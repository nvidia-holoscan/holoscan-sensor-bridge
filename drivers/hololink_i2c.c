/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
 *
 * See README.md for detailed information.
 */

#include <linux/module.h>
#include <linux/i2c.h>
#include <linux/kernel.h>
#include <linux/of.h>
#include <linux/platform_device.h>

#include "trace.h"

uint count = 1;
module_param(count, uint, 0644);
MODULE_PARM_DESC(count, "Number of I2C busses to create.");

struct hololink_i2c_dev {
    dev_t devt;
    struct class *class;
    struct hololink_i2c_bus_dev **bus_list;
    uint busses;
};

struct hololink_i2c_bus_dev {
    struct cdev cdev;
    struct device *dev;
    struct i2c_adapter adapter;
    struct hololink_i2c_file *reader;
};

struct hololink_i2c_file {
    struct hololink_i2c_bus_dev *hdev;
    __u8 buffer[8192];
    size_t buffer_size;
    struct mutex read_lock;
    struct mutex write_lock;
    __u16 transfer_key;
};

#define HOLOLINK_I2C_TRANSFER 1
#define HOLOLINK_I2C_OPEN 2

static struct hololink_i2c_dev *hololink_i2c_dev;

static inline int add_header(__u16 * p, size_t limit, size_t * index, struct i2c_msg * msg)
{
    // limit and index are in terms of 16-bit words
    size_t n = index[0];
    if ((n + 3) > limit) {
        return 0;
    }
    p[n++] = msg->addr;
    p[n++] = msg->flags;
    p[n++] = msg->len;
    index[0] = n;
    return 1;
}

static inline int add_buffer(unsigned char * p, size_t limit, size_t * index, struct i2c_msg * msg)
{
    // limit and index are in terms of 8-bit bytes
    size_t n = index[0];
    // Only add buffers on write requests.
    if (msg->flags & I2C_M_RD) {
        return 1;
    }
    if ((n + msg->len) > limit) {
        return 0;
    }
    memcpy(&p[n], msg->buf, msg->len);
    index[0] = n + msg->len;
    return 1;
}

static inline size_t serialize_msgs(struct hololink_i2c_file *hlf, unsigned char * buffer, size_t limit, int command, struct i2c_msg msgs[], int num)
{
    size_t index;
    int i;
    // Save headers first.
    __u16 * p = (__u16*) buffer;
    size_t p_limit = limit / sizeof(p[0]);
    size_t p_index = 0;
    __u16 transfer_key = ++hlf->transfer_key;

    // Add the msgs count.
    p[p_index++] = HOLOLINK_I2C_TRANSFER;
    p[p_index++] = transfer_key;
    p[p_index++] = num;
    // Add headers.
    for (i = 0; i < num; i++) {
        if (! add_header(p, p_limit, &p_index, &msgs[i])) {
            WARNING("I2C transaction overflow i=%d num=%d.", i, num);
            return 0;
        }
    }
    // Add write blobs.
    index = p_index * sizeof(p[0]);
    for (i = 0; i < num; i++) {
        if (! add_buffer(buffer, limit, &index, &msgs[i])) {
            WARNING("I2C transaction buffer overflow i=%d num=%d.", i, num);
            return 0;
        }
    }
    return index;
}

static int hololink_i2c_xfer(struct i2c_adapter *adap, struct i2c_msg msgs[], int num)
{
    struct hololink_i2c_bus_dev *hdev = container_of(adap, struct hololink_i2c_bus_dev, adapter);
    struct hololink_i2c_file *hlf = hdev->reader;
    int message_count, j;
    struct i2c_msg * m;
    __u16 * reply;
    unsigned reply_words;

    TRACE("hdev=%p hlf=%p num=%d.", hdev, hlf, num);

    if (!hlf) {
        WARNING("No supporting user mode connection; ignoring.");
        return 0;
    }

    if (hlf->buffer_size) {
        WARNING("I2C transaction steps on a queued transaction; ignoring.");
        return 0;
    }

    // Serialize commands and written data
    hlf->buffer_size = serialize_msgs(hlf, hlf->buffer, sizeof(hlf->buffer), HOLOLINK_I2C_TRANSFER, msgs, num);
    if (hlf->buffer_size != 0) {
        // Our egress buffer is set up; wake up the read size which
        // will pass this up to user-mode
        mutex_unlock(&hlf->read_lock);
    }
    // Wait for the result to be written.
    TRACE("Waiting for reply.");
    mutex_lock(&hlf->write_lock);
    // Gather the read data and pass it back to the xfer request.
    reply = (__u16*) hlf->buffer;
    reply_words = hlf->buffer_size / sizeof(reply[0]);
    j = 0;
    if (reply_words < 2) {
        WARNING("Reply of %u bytes is too small.", (unsigned) hlf->buffer_size);
    } else if (reply[j++] != HOLOLINK_I2C_TRANSFER) {
        WARNING("Unexpected command in reply (%u).", (unsigned) reply[0]);
    } else if (reply[j++] != hlf->transfer_key) {
        WARNING("Unexpected key in reply (%u).", (unsigned) reply[1]);
    } else {
        j *= sizeof(reply[0]);
        for (message_count = 0; message_count < (unsigned)num; message_count++) {
            m = &(msgs[message_count]);
            if ((m->flags & I2C_M_RD) == 0) {
                continue;
            }
            if ((j + m->len) > sizeof(hlf->buffer)) {
                WARNING("Unexpected size in message %d.", message_count);
                return -EINVAL;
            }
            memcpy(m->buf, &(hlf->buffer[j]), m->len);
            j += m->len;
        }
    }

    hlf->buffer_size = 0;
    TRACE("continuing.");
    return message_count;
}

static u32 hololink_i2c_functionality(struct i2c_adapter *adap)
{
    u32 ret = I2C_FUNC_I2C | (I2C_FUNC_SMBUS_EMUL & ~I2C_FUNC_SMBUS_QUICK);
    return ret;
}

static const struct i2c_algorithm hololink_i2c_algo = {
    .master_xfer = hololink_i2c_xfer,
    .functionality = hololink_i2c_functionality,
};

static int hololink_i2c_open(struct inode *inode, struct file *filp)
{
    struct hololink_i2c_file *hlf;
    struct hololink_i2c_bus_dev *hdev = container_of(inode->i_cdev, struct hololink_i2c_bus_dev, cdev);
    struct device *dev = hdev->dev;
    __u16 * u16_buffer;
    unsigned n;

    TRACE("inode=%p filp=%p hdev=%p dev=%p.", inode, filp, hdev, dev);

    hlf = kzalloc(sizeof(*hlf), GFP_KERNEL);
    if (IS_ERR(hlf)) {
        printk(KERN_INFO "hololink_i2c_open failed to allocate memory.\n");
        return -ENOMEM;
    }
    hlf->hdev = hdev;

    // Start read_lock and write_lock in the locked state.
    mutex_init(&hlf->read_lock);
    mutex_lock(&hlf->read_lock);
    mutex_init(&hlf->write_lock);
    mutex_lock(&hlf->write_lock);

    // Initialize the buffer we'll pass back on the first read.
    u16_buffer = (__u16*) &(hlf->buffer);
    n = 0;
    u16_buffer[n++] = HOLOLINK_I2C_OPEN;
    u16_buffer[n++] = hdev->adapter.nr;
    hlf->buffer_size = n * sizeof(u16_buffer[0]);
    mutex_unlock(&hlf->read_lock);

    hdev->reader = hlf;
    filp->private_data = hlf;
    return 0;
}

static int hololink_i2c_release(struct inode *inode, struct file *filp)
{
    struct hololink_i2c_file *hlf = filp->private_data;
    struct hololink_i2c_bus_dev *hdev = hlf->hdev;

    TRACE("inode=%p filp=%p hlf=%p.", inode, filp, hlf);
    hdev->reader = NULL;

    mutex_unlock(&hlf->read_lock);
    mutex_unlock(&hlf->write_lock);

    mutex_destroy(&hlf->read_lock);
    mutex_destroy(&hlf->write_lock);
    kfree(hlf);
    return 0;
}

static ssize_t hololink_i2c_read(struct file *filp, char __user *buf, size_t size, loff_t *pos)
{
    struct hololink_i2c_file *hlf = filp->private_data;
    int ret;
    size_t bytes = size;
    TRACE("filp=%p buf=%p size=%lu pos[0]=%d hlf=%p.", filp, buf, (unsigned long) size, (int) pos[0], hlf);
    ret = mutex_lock_interruptible(&hlf->read_lock);
    if (ret) {
        DEBUG("interrupted, ret=%d.", ret);
        if (ret > 0) {
            // I DONT THINK THIS HAPPENS BUT BE SURE
            // we don't want an error here to be
            // mistaken as a valid read.
            ret = -ret;
        }
        return ret;
    }
    TRACE("bytes=%u hlf->buffer_size=%u.", (unsigned) bytes, (unsigned) hlf->buffer_size);
    if (bytes > hlf->buffer_size) {
        bytes = hlf->buffer_size;
    }
    if (copy_to_user(buf, &hlf->buffer, bytes)) {
        return -EFAULT;
    }
    hlf->buffer_size = 0;
    return bytes;
}

static ssize_t hololink_i2c_write(struct file *filp, const char __user *buf, size_t size, loff_t *pos)
{
    struct hololink_i2c_file *hlf = filp->private_data;
    size_t bytes = sizeof(hlf->buffer);

    TRACE("filp=%p buf=%p size=%lu pos[0]=%d hlf=%p.", filp, buf, (unsigned long) size, (int) pos[0], hlf);

    if (hlf->buffer_size) {
        WARNING("Ignoring write of %u bytes; buffer_size=%u.",
            (unsigned) size, (unsigned) hlf->buffer_size);
        return 0;
    }
    if (bytes < size) {
        WARNING("Ignoring write of %u bytes; size limit=%u.",
            (unsigned) size, (unsigned) bytes);
        return 0;
    }
    if (bytes > size) {
        bytes = size;
    }
    if (copy_from_user(hlf->buffer, buf, bytes)) {
        return -EFAULT;
    }
    hlf->buffer_size = bytes;
    mutex_unlock(&hlf->write_lock);
    return bytes;
}

long hololink_i2c_unlocked_ioctl(struct file *filp, u_int cmd, u_long arg)
{
    long ret = -EINVAL;
    struct hololink_i2c_file *hlf = filp->private_data;
    struct hololink_i2c_bus_dev * hdev = hlf->hdev;
    void __user *argp = (void __user *)arg;

    TRACE("filp=%p cmd=0x%X arg=0x%lX hlf=%p hdev=%p argp=%p.", filp, cmd, arg, hlf, hdev, argp);

    return ret;
}

static const struct file_operations hololink_i2c_file_ops = {
    .owner = THIS_MODULE,
    .open = hololink_i2c_open,
    .release = hololink_i2c_release,
    .read = hololink_i2c_read,
    .write = hololink_i2c_write,
    .unlocked_ioctl = hololink_i2c_unlocked_ioctl,
};

static int __init hololink_i2c_init(void)
{
    int ret;
    struct hololink_i2c_bus_dev *i2c_dev;
    struct hololink_i2c_dev *hdev;
    dev_t devt;

    if (count > 100) {
        WARNING("Cowardly refusing to make %u busses.", (unsigned) count);
        return -EINVAL;
    }

    hdev = (struct hololink_i2c_dev*) kzalloc(sizeof(*hololink_i2c_dev), GFP_KERNEL);
    if (! hdev) {
        WARNING("Can't allocate memory for hololink_i2c_dev.");
        return -ENOMEM;
    }

    hdev->bus_list = kzalloc(sizeof(*hdev->bus_list) * count, GFP_KERNEL);
    if (! hdev->bus_list) {
        WARNING("Can't allocate memory for bus_list, count=%u.", (unsigned) count);
        kfree(hdev);
        return -ENOMEM;
    }

    hololink_i2c_dev = hdev;
    ret = alloc_chrdev_region(&hdev->devt, 0, count, "hololink_i2c");
    TRACE("alloc_chrdev_region ret=%d.", ret);
    if (ret < 0) {
        printk(KERN_INFO "hololink_i2c_probe alloc_chrdev_region ret=%d.\n", ret);
        goto cleanup;
    }
    TRACE("major=%d minor=%d", MAJOR(hdev->devt), MINOR(hdev->devt));

    hdev->class = class_create(THIS_MODULE, "hololink_i2c");
    if (IS_ERR(hdev->class)) {
        printk(KERN_INFO "hololink_i2c_probe unable to register a class.\n");
        goto cleanup;
    }

    for (hdev->busses = 0; hdev->busses < count; hdev->busses++) {
        i2c_dev = kzalloc(sizeof(*i2c_dev), GFP_KERNEL);
        if (i2c_dev == NULL) {
            WARNING("Can't allocate memory for bus=%u.", (unsigned) hdev->busses);
            ret = -ENOMEM;
            goto cleanup;
        }

        devt = MKDEV(MAJOR(hdev->devt), MINOR(hdev->devt)+hdev->busses);
        cdev_init(&i2c_dev->cdev, &hololink_i2c_file_ops);
        cdev_add(&i2c_dev->cdev, devt, 1);
        TRACE("cdev=%p", &i2c_dev->cdev);
        i2c_dev->dev = device_create(hdev->class, NULL, devt, NULL, "%s%u", "hololink_i2c", (unsigned) hdev->busses);

        i2c_set_adapdata(&i2c_dev->adapter, i2c_dev);
        i2c_dev->adapter.owner = THIS_MODULE;
        i2c_dev->adapter.class = I2C_CLASS_DEPRECATED;
        i2c_dev->adapter.algo = &hololink_i2c_algo;
        snprintf(i2c_dev->adapter.name, sizeof(i2c_dev->adapter.name), "hololink_i2c%d", hdev->busses);
        i2c_dev->adapter.nr = -1;  // Pick the next one.

        ret = i2c_add_adapter(&i2c_dev->adapter);
        DEBUG("i2c_add_adapter adapter=%p nr=%d ret=%d.", &(i2c_dev->adapter), (int) i2c_dev->adapter.nr, ret);
        if (ret < 0) {
            kfree(i2c_dev);
            goto cleanup;
        }

        hdev->bus_list[hdev->busses] = i2c_dev;
    }
    TRACE("done.");
    return ret;
cleanup:
    hololink_i2c_dev = NULL;
    while (hdev->busses--) {
        struct hololink_i2c_bus_dev *i2c_dev = hdev->bus_list[hdev->busses];
        hdev->bus_list[hdev->busses] = NULL;
        i2c_del_adapter(&i2c_dev->adapter);
        devt = MKDEV(MAJOR(hdev->devt), MINOR(hdev->devt)+hdev->busses);
        device_destroy(hdev->class, devt);
        cdev_del(&i2c_dev->cdev);
        kfree(i2c_dev);
    }
    if (hdev->devt) {
        unregister_chrdev_region(hdev->devt, count);
    }
    class_destroy(hdev->class);
    kfree(hdev->bus_list);
    kfree(hdev);
    return ret;
}

module_init(hololink_i2c_init);

static void __exit hololink_i2c_exit(void)
{
    struct hololink_i2c_dev *hdev = hololink_i2c_dev;
    dev_t devt;

    TRACE("exit.");
    hololink_i2c_dev = NULL;
    while (hdev->busses--) {
        struct hololink_i2c_bus_dev *i2c_dev = hdev->bus_list[hdev->busses];
        hdev->bus_list[hdev->busses] = NULL;
        i2c_del_adapter(&i2c_dev->adapter);
        devt = MKDEV(MAJOR(hdev->devt), MINOR(hdev->devt)+hdev->busses);
        device_destroy(hdev->class, devt);
        cdev_del(&i2c_dev->cdev);
        kfree(i2c_dev);
    }
    unregister_chrdev_region(hdev->devt, count);
    class_destroy(hdev->class);
    kfree(hdev->bus_list);
    kfree(hdev);
}
module_exit(hololink_i2c_exit);

MODULE_DESCRIPTION("NVIDIA Hololink I2C Bus Controller driver.");
MODULE_AUTHOR("Patrick O'Grady");
MODULE_LICENSE("GPL v2");
