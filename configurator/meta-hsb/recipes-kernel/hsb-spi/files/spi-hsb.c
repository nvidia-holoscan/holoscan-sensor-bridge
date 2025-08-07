/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/cdev.h>
#include <linux/slab.h>
#include <linux/spi/spi.h>
#include <linux/platform_device.h>
#ifdef ENABLE_JESD
#include <linux/jesd204/jesd204.h>
#endif

uint max_hsb_devices = 4;
module_param(max_hsb_devices, uint, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
MODULE_PARM_DESC(max_hsb_devices, "Maximum number of HSB devices");

uint max_spi_devices = 4;
module_param(max_spi_devices, uint, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
MODULE_PARM_DESC(max_spi_devices, "Maximum number of SPI devices per HSB");

// Per-module state.
static struct {
    int num_devices;
    dev_t devt;
    struct class *class;
} hsb_spi_module = {0};

// Per-device state.
struct hsb_spi_device {
    dev_t devt;
    struct cdev cdev;
    struct device *dev;
    struct hsb_spi_file *file;
#ifdef ENABLE_JESD
    struct jesd204_dev *jdev;
#endif
};

/*******************************************************************************
 * Character device functions (for communication with userspace client/daemon).
*******************************************************************************/

// Messages are either SPI transactions or callbacks for the JESD204 FSM.
#define HSB_SPI_MSG_TYPE_SPI 0
#define HSB_SPI_MSG_TYPE_JESD 1

// Maximum transfer size.
#define BUFFER_SIZE (256)

// Message passed between this driver and the userspace client.
struct hsb_spi_message {
    uint8_t type;
    union {
        struct {
            uint8_t cs;
            uint8_t cmd_bytes;
            uint8_t wr_bytes;
            uint8_t rd_bytes;
        } spi;
        struct {
            uint8_t id;
        } jesd;
    } u;
};

struct hsb_spi_file {
    struct hsb_spi_device *hsb_dev;
    wait_queue_head_t queue;
    __u8 buffer[BUFFER_SIZE];
    size_t bytes_available;
    size_t bytes_written;
};

static int hsb_spi_open(struct inode *inode, struct file *filep)
{
    struct hsb_spi_file *hsb_file;
    struct hsb_spi_device *hsb_dev = container_of(inode->i_cdev, struct hsb_spi_device, cdev);

    pr_info("hsb_spi_open: opened device /dev/%s\n", filep->f_path.dentry->d_name.name);

    if (hsb_dev->file) {
        pr_err("file already open\n");
        return -EBUSY;
    }

    hsb_file = kzalloc(sizeof(*hsb_file), GFP_KERNEL);
    if (IS_ERR(hsb_file)) {
        pr_err("hsb_spi_open failed to allocate memory\n");
        return PTR_ERR(hsb_file);
    }

    hsb_file->hsb_dev = hsb_dev;
    init_waitqueue_head(&hsb_file->queue);

    hsb_dev->file = hsb_file;
    filep->private_data = hsb_file;

    return 0;
}

static int hsb_spi_release(struct inode *inode, struct file *filep)
{
    struct hsb_spi_file *hsb_file = filep->private_data;
    struct hsb_spi_device *hsb_dev = hsb_file->hsb_dev;

    pr_info("hsb_spi_release: closed device /dev/%s\n", filep->f_path.dentry->d_name.name);

    hsb_dev->file = NULL;
    kfree(hsb_file);

    return 0;
}

static ssize_t hsb_spi_read(struct file *filep, char __user *buf, size_t size, loff_t *pos)
{
    struct hsb_spi_file *hsb_file = filep->private_data;
    size_t bytes_read;

    wait_event_interruptible(hsb_file->queue, hsb_file->bytes_available > 0);
    if (size < hsb_file->bytes_available) {
        pr_err("output buffer too small (has %zu, provided %zu)\n", size, hsb_file->bytes_available);
        return -EFAULT;
    }

    if (copy_to_user(buf, &hsb_file->buffer, hsb_file->bytes_available)) {
        return -EFAULT;
    }
    bytes_read = hsb_file->bytes_available;
    hsb_file->bytes_available = 0;

    wake_up_interruptible(&hsb_file->queue);

    return bytes_read;
}

static ssize_t hsb_spi_write(struct file *filep, const char __user *buf, size_t size, loff_t *pos)
{
    struct hsb_spi_file *hsb_file = filep->private_data;

    if (hsb_file->bytes_available || hsb_file->bytes_written) {
        pr_err("Data still pending\n");
        return -EFAULT;
    }
    if (sizeof(hsb_file->buffer) < size) {
        pr_err("data size exceeds limit (has %zu, provided %zu)\n", sizeof(hsb_file->buffer), size);
        return -EFAULT;
    }

    if (copy_from_user(hsb_file->buffer, buf, size)) {
        return -EFAULT;
    }
    hsb_file->bytes_written = size;

    wake_up_interruptible(&hsb_file->queue);

    return size;
}

static const struct file_operations hsb_spi_file_ops = {
    .owner = THIS_MODULE,
    .open = hsb_spi_open,
    .release = hsb_spi_release,
    .read = hsb_spi_read,
    .write = hsb_spi_write,
};

/*******************************************************************************
 * JESD204 functions (sends JESD state transition callbacks to client).
*******************************************************************************/

#ifdef ENABLE_JESD
struct hsb_jesd204_priv {
    struct hsb_spi_device *hsb_dev;
};

static int hsb_spi_jesd_transition(struct hsb_spi_device *hsb_dev, const char* name, uint8_t id) {
    struct hsb_spi_file *hsb_file = hsb_dev->file;
    struct hsb_spi_message *hsb_message = (struct hsb_spi_message*)hsb_file->buffer;
    uint8_t response = 0;

    wait_event_interruptible(hsb_file->queue, hsb_file->bytes_available == 0);

    // Send the message for the callback.
    hsb_message->type = HSB_SPI_MSG_TYPE_JESD;
    hsb_message->u.jesd.id = id;
    hsb_file->bytes_available = sizeof(*hsb_message);

    wake_up_interruptible(&hsb_file->queue);

    // Wait for the response.
    wait_event_interruptible(hsb_file->queue, hsb_file->bytes_written > 0);
    if (hsb_file->bytes_written != 1) {
        pr_err("JESD transition response expected 1 byte, got %zu\n",
                hsb_file->bytes_written);
        return -EFAULT;
    }
    response = *((uint8_t*)hsb_file->buffer);
    hsb_file->bytes_written = 0;

    // Note: We don't currently handle a STATE_CHANGE_DEFER response and
    //       always assume STATE_CHANGE_DONE.

    dev_dbg(hsb_dev->dev, "JESD transition to state %u, %s\n", id, name); \

    return 0;
}

#define JESD204_FUNC(name) \
static int func_##name(struct jesd204_dev *jdev, enum jesd204_state_op_reason reason) { \
    struct hsb_jesd204_priv *jpriv = jesd204_dev_priv(jdev); \
    hsb_spi_jesd_transition(jpriv->hsb_dev, #name, name); \
    return JESD204_STATE_CHANGE_DONE; \
}

JESD204_FUNC(JESD204_OP_DEVICE_INIT);
JESD204_FUNC(JESD204_OP_LINK_INIT);
JESD204_FUNC(JESD204_OP_LINK_SUPPORTED);
JESD204_FUNC(JESD204_OP_LINK_PRE_SETUP);
JESD204_FUNC(JESD204_OP_CLK_SYNC_STAGE1);
JESD204_FUNC(JESD204_OP_CLK_SYNC_STAGE2);
JESD204_FUNC(JESD204_OP_CLK_SYNC_STAGE3);
JESD204_FUNC(JESD204_OP_LINK_SETUP);
JESD204_FUNC(JESD204_OP_OPT_SETUP_STAGE1);
JESD204_FUNC(JESD204_OP_OPT_SETUP_STAGE2);
JESD204_FUNC(JESD204_OP_OPT_SETUP_STAGE3);
JESD204_FUNC(JESD204_OP_OPT_SETUP_STAGE4);
JESD204_FUNC(JESD204_OP_OPT_SETUP_STAGE5);
JESD204_FUNC(JESD204_OP_CLOCKS_ENABLE);
JESD204_FUNC(JESD204_OP_LINK_ENABLE);
JESD204_FUNC(JESD204_OP_LINK_RUNNING);
JESD204_FUNC(JESD204_OP_OPT_POST_RUNNING_STAGE);

static const struct jesd204_dev_data jesd204_hsb_init = {
	.state_ops = {
		[JESD204_OP_DEVICE_INIT] = {
			.per_device = func_JESD204_OP_DEVICE_INIT,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_LINK_INIT] = {
			.per_device = func_JESD204_OP_LINK_INIT,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_LINK_SUPPORTED] = {
			.per_device = func_JESD204_OP_LINK_SUPPORTED,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_LINK_PRE_SETUP] = {
			.per_device = func_JESD204_OP_LINK_PRE_SETUP,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_CLK_SYNC_STAGE1] = {
			.per_device = func_JESD204_OP_CLK_SYNC_STAGE1,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_CLK_SYNC_STAGE2] = {
			.per_device = func_JESD204_OP_CLK_SYNC_STAGE2,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_CLK_SYNC_STAGE3] = {
			.per_device = func_JESD204_OP_CLK_SYNC_STAGE3,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_LINK_SETUP] = {
			.per_device = func_JESD204_OP_LINK_SETUP,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_OPT_SETUP_STAGE1] = {
			.per_device = func_JESD204_OP_OPT_SETUP_STAGE1,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_OPT_SETUP_STAGE2] = {
			.per_device = func_JESD204_OP_OPT_SETUP_STAGE2,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_OPT_SETUP_STAGE3] = {
			.per_device = func_JESD204_OP_OPT_SETUP_STAGE3,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_OPT_SETUP_STAGE4] = {
			.per_device = func_JESD204_OP_OPT_SETUP_STAGE4,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_OPT_SETUP_STAGE5] = {
			.per_device = func_JESD204_OP_OPT_SETUP_STAGE5,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_CLOCKS_ENABLE] = {
			.per_device = func_JESD204_OP_CLOCKS_ENABLE,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_LINK_ENABLE] = {
			.per_device = func_JESD204_OP_LINK_ENABLE,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_LINK_RUNNING] = {
			.per_device = func_JESD204_OP_LINK_RUNNING,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
		[JESD204_OP_OPT_POST_RUNNING_STAGE] = {
			.per_device = func_JESD204_OP_OPT_POST_RUNNING_STAGE,
            .mode = JESD204_STATE_OP_MODE_PER_DEVICE,
		},
	},
	.max_num_links = 4,
	.num_retries = 3,
	.sizeof_priv = sizeof(struct hsb_jesd204_priv),
};
#endif // ENABLE_JESD

/*******************************************************************************
 * SPI controller functions.
*******************************************************************************/

static void dbg_data(struct device *dev,
                     const uint8_t* tx, size_t tx_len,
                     const uint8_t* rx, size_t rx_len) {
#ifdef DEBUG
    int i;
    char str[256];

    sprintf(&str[strlen(str)], "[");
    for (i = 0; i < tx_len; i++) {
        if (i != 0) {
            sprintf(&str[strlen(str)], ", ");
        }
        sprintf(&str[strlen(str)], "%02x", tx[i]);
    }
    sprintf(&str[strlen(str)], "]");

    if (rx && (tx[0] & 0x80)) {
        sprintf(&str[strlen(str)], " --> [");
        for (i = 0; i < rx_len; i++) {
            if (i != 0) {
                sprintf(&str[strlen(str)], ", ");
            }
            sprintf(&str[strlen(str)], "%02x", rx[i]);
        }
        sprintf(&str[strlen(str)], "]");
    }
    dev_dbg(dev, "spi xfer %s\n", str);
#endif
}

static int hsb_spi_transfer_one_message(struct spi_controller *ctlr,
                                        struct spi_message *msg) {
    struct hsb_spi_device *hsb_dev = spi_controller_get_devdata(ctlr);
    struct hsb_spi_file *hsb_file = hsb_dev->file;
    struct hsb_spi_message *hsb_message = (struct hsb_spi_message*)hsb_file->buffer;
    struct spi_transfer *xfer, *next;
    size_t hsb_message_size;
    int ret = 0;

    msg->status = 0;
    msg->actual_length = 0;

    list_for_each_entry(xfer, &msg->transfers, transfer_list) {
        if (xfer->len == 0) {
            continue;
        }

        hsb_message_size = sizeof(*hsb_message) + xfer->len;
        if (hsb_message_size > sizeof(hsb_file->buffer)) {
            pr_err("SPI message exceeds limit (has %zu, provided %zu)\n",
                   sizeof(hsb_file->buffer), hsb_message_size);
            ret = -EFAULT;
            goto done;
        }

        if (xfer->tx_buf) {
            // If both TX and RX buffers given in a single transfer, it's a CMD packet.
            if (xfer->rx_buf) {
                wait_event_interruptible(hsb_file->queue, hsb_file->bytes_available == 0);

                // Send the write message.
                hsb_message->type = HSB_SPI_MSG_TYPE_SPI;
                hsb_message->u.spi.cs = spi_get_chipselect(msg->spi, 0);
                hsb_message->u.spi.cmd_bytes = xfer->len;
                hsb_message->u.spi.wr_bytes = xfer->len;
                hsb_message->u.spi.rd_bytes = 0;
                memcpy(hsb_file->buffer + sizeof(*hsb_message), xfer->tx_buf, xfer->len);
                hsb_file->bytes_available = hsb_message_size;

                wake_up_interruptible(&hsb_file->queue);

                // Wait for the response.
                wait_event_interruptible(hsb_file->queue, hsb_file->bytes_written > 0);
                if (hsb_file->bytes_written != xfer->len) {
                    pr_err("SPI response size mismatch (got %zu, expected %zu)\n",
                            hsb_file->bytes_written, xfer->len);
                    ret = -EFAULT;
                    goto done;
                }
                memcpy(xfer->rx_buf, hsb_file->buffer, hsb_file->bytes_written);
                hsb_file->bytes_written = 0;

                dbg_data(hsb_dev->dev, xfer->tx_buf, xfer->len, xfer->rx_buf, xfer->len);
            } else {
                // If just a TX buffer is given in this transfer and the next transfer
                // is just an RX buffer, then it's a WR_RD packet.
                next = list_next_entry(xfer, transfer_list);
                if (!list_entry_is_head(next, &msg->transfers, transfer_list) &&
                    !next->tx_buf && next->rx_buf) {
                    wait_event_interruptible(hsb_file->queue, hsb_file->bytes_available == 0);

                    // Send the write message.
                    hsb_message->type = HSB_SPI_MSG_TYPE_SPI;
                    hsb_message->u.spi.cs = spi_get_chipselect(msg->spi, 0);
                    hsb_message->u.spi.cmd_bytes = 0;
                    hsb_message->u.spi.wr_bytes = xfer->len;
                    hsb_message->u.spi.rd_bytes = next->len;
                    memcpy(hsb_file->buffer + sizeof(*hsb_message), xfer->tx_buf, xfer->len);
                    hsb_file->bytes_available = hsb_message_size;

                    wake_up_interruptible(&hsb_file->queue);

                    // Wait for the response.
                    wait_event_interruptible(hsb_file->queue, hsb_file->bytes_written > 0);
                    if (hsb_file->bytes_written != next->len) {
                        pr_err("SPI response size mismatch (got %zu, expected %zu)\n",
                                hsb_file->bytes_written, next->len);
                        ret = -EFAULT;
                        goto done;
                    }
                    memcpy(next->rx_buf, hsb_file->buffer, hsb_file->bytes_written);
                    hsb_file->bytes_written = 0;

                    dbg_data(hsb_dev->dev, xfer->tx_buf, xfer->len, next->rx_buf, next->len);

                    xfer = next;
                } else {
                    // Otherwise, just a TX buffer is a WR packet.

                    wait_event_interruptible(hsb_file->queue, hsb_file->bytes_available == 0);

                    // Send the write message.
                    hsb_message->type = HSB_SPI_MSG_TYPE_SPI;
                    hsb_message->u.spi.cs = spi_get_chipselect(msg->spi, 0);
                    hsb_message->u.spi.cmd_bytes = 0;
                    hsb_message->u.spi.wr_bytes = xfer->len;
                    hsb_message->u.spi.rd_bytes = 0;
                    memcpy(hsb_file->buffer + sizeof(*hsb_message), xfer->tx_buf, xfer->len);
                    hsb_file->bytes_available = hsb_message_size;

                    wake_up_interruptible(&hsb_file->queue);

                    dbg_data(hsb_dev->dev, xfer->tx_buf, xfer->len, NULL, 0);
                }
            }
        } else if (xfer->rx_buf) {
            pr_err(" Read-only SPI transactions currently not supported.");
            return EPERM;
        }

        msg->actual_length += xfer->len;
    }

done:
    spi_finalize_current_message(ctlr);

    return ret;
}

/*******************************************************************************
 * Module functions.
*******************************************************************************/

static int hsb_spi_module_init(void)
{
    int ret;

    ret = alloc_chrdev_region(&hsb_spi_module.devt, 0, max_hsb_devices, "hsbspi");
    if (ret < 0) {
        pr_err("alloc_chrdev_region failed, ret=%d\n", ret);
        return ret;
    }

    hsb_spi_module.class = class_create(THIS_MODULE, "hsbspi");
    if (IS_ERR(hsb_spi_module.class)) {
        ret = PTR_ERR(hsb_spi_module.class);
        pr_err("class_create failed, ret=%d\n", ret);
        unregister_chrdev_region(hsb_spi_module.devt, max_hsb_devices);
        return ret;
    }

    return 0;
}

static void hsb_spi_module_exit(void)
{
    unregister_chrdev_region(hsb_spi_module.devt, max_hsb_devices);
    class_destroy(hsb_spi_module.class);
}

static int hsb_spi_probe(struct platform_device *pdev)
{
    struct spi_controller *ctlr;
    struct hsb_spi_device *hsb_dev;
    int ret;
#ifdef ENABLE_JESD
    struct jesd204_dev *jdev;
    struct hsb_jesd204_priv *jpriv;
#endif

    if (hsb_spi_module.num_devices == 0) {
        ret = hsb_spi_module_init();
        if (ret < 0) {
            dev_err(&pdev->dev, "hsb_spi_module_init failed, ret=%d\n", ret);
            return ret;
        }
    }

    pdev->id = of_alias_get_id(pdev->dev.of_node, "spi");
    if (pdev->id < 0) {
        dev_err(&pdev->dev, "of_alias_get_id failed, ret=%d\n", ret);
        return ret;
    }

    ctlr = spi_alloc_master(&pdev->dev, sizeof(*hsb_dev));
    if (!ctlr) {
        ret = -ENOMEM;
        dev_err(&pdev->dev, "spi controller alloc failed\n");
        goto cleanup;
    }
    platform_set_drvdata(pdev, ctlr);
    hsb_dev = spi_controller_get_devdata(ctlr);

    ctlr->max_speed_hz = 25000000; /* 25MHz */
    ctlr->bits_per_word_mask = SPI_BPW_RANGE_MASK(4, 32);
    ctlr->transfer_one_message = hsb_spi_transfer_one_message;
    ctlr->num_chipselect = max_spi_devices;
    ctlr->bus_num = pdev->id;

    ctlr->dev.of_node = pdev->dev.of_node;
    ret = devm_spi_register_controller(&pdev->dev, ctlr);
    if (ret < 0) {
        dev_err(&pdev->dev, "spi controller register failed, ret=%d\n", ret);
        goto cleanup;
    }

    // Create device.
    hsb_dev->devt = MKDEV(MAJOR(hsb_spi_module.devt), MINOR(hsb_spi_module.devt) + pdev->id);
    cdev_init(&hsb_dev->cdev, &hsb_spi_file_ops);
    ret = cdev_add(&hsb_dev->cdev, hsb_dev->devt, 1);
    if (ret < 0) {
        dev_err(&pdev->dev, "cdev_add failed, ret=%d\n", ret);
        goto cleanup;
    }

    hsb_dev->dev = device_create(hsb_spi_module.class, NULL, hsb_dev->devt, NULL,
                                 "%s%u", "hsbspi", (unsigned)pdev->id);
    if (IS_ERR(hsb_dev->dev)) {
        ret = PTR_ERR(hsb_dev->dev);
        dev_err(&pdev->dev, "device_create failed, ret=%d", ret);
        goto cleanup;
    }

#ifdef ENABLE_JESD
    jdev = devm_jesd204_dev_register(&pdev->dev, &jesd204_hsb_init);
    if (IS_ERR(jdev) || !jdev) {
        ret = jdev ? PTR_ERR(jdev) : -ENODEV;
    } else {
        hsb_dev->jdev = jdev;

        jpriv = jesd204_dev_priv(jdev);
        jpriv->hsb_dev = hsb_dev;

        ret = jesd204_fsm_start(jdev, JESD204_LINKS_ALL);
        if (ret < 0) {
            dev_err(&pdev->dev, "jesd204_fsm_start failed, ret=%d\n", ret);
            goto cleanup;
        }
    }
#endif

    hsb_spi_module.num_devices++;

    pr_info("hsb_spi_probe: created device /dev/hsbspi%u\n", (unsigned)pdev->id);

    return 0;

cleanup:
    device_destroy(hsb_spi_module.class, hsb_dev->devt);
    cdev_del(&hsb_dev->cdev);
    spi_controller_put(ctlr);

    if (hsb_spi_module.num_devices == 0) {
        hsb_spi_module_exit();
    }

    return ret;
}

static int hsb_spi_remove(struct platform_device *pdev) {
    struct spi_controller *ctlr = platform_get_drvdata(pdev);
    struct hsb_spi_device *hsb_dev = spi_controller_get_devdata(ctlr);

#ifdef ENABLE_JESD
    if (hsb_dev->jdev) {
        jesd204_fsm_stop(hsb_dev->jdev, JESD204_LINKS_ALL);
    }
#endif

    device_destroy(hsb_spi_module.class, hsb_dev->devt);
    cdev_del(&hsb_dev->cdev);

    hsb_spi_module.num_devices--;
    if (hsb_spi_module.num_devices == 0) {
        hsb_spi_module_exit();
    }

    pr_info("hsb_spi_probe: removed device /dev/hsbspi%u\n", (unsigned)pdev->id);

    return 0;
}

static const struct of_device_id hsb_spi_of_match[] = {
    { .compatible = "nvidia,hsb-spi" },
    {}
};
MODULE_DEVICE_TABLE(of, hsb_spi_of_match);

static struct platform_driver hsb_spi_driver = {
    .driver = {
        .name   = "spi-hsb",
        .of_match_table = hsb_spi_of_match,
    },
    .probe      = hsb_spi_probe,
    .remove     = hsb_spi_remove,
};
module_platform_driver(hsb_spi_driver);

MODULE_ALIAS("platform:spi-hsb");
MODULE_DESCRIPTION("NVIDIA Holoscan Sensor Bridge SPI Controller Driver");
MODULE_AUTHOR("Ian Stewart <istewart@nvidia.com>");
MODULE_LICENSE("GPL v2");
