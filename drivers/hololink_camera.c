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
#include <linux/kernel.h>
#include <linux/platform_device.h>
#include <linux/videodev2.h>
#include <media/v4l2-ctrls.h>
#include <media/v4l2-device.h>
#include <media/v4l2-event.h>
#include <media/v4l2-ioctl.h>
#include <media/videobuf2-v4l2.h>
#include <media/videobuf2-dma-contig.h>

#include "trace.h"

static int i2c_bus_number;
module_param(i2c_bus_number, int, 0644);
MODULE_PARM_DESC(i2c_bus_number, "I2C Bus number where this device is connected.");

static int i2c_device_address;
module_param(i2c_device_address, int, 0644);
MODULE_PARM_DESC(i2c_device_address, "I2C Device address.");

struct hololink_camera_dev {
    struct platform_device *pdev;
    struct device *dev;
    struct v4l2_async_notifier v4l2_notifier;
    struct v4l2_device v4l2_dev;
    struct v4l2_subdev *v4l2_subdev;
    struct video_device video_dev;
    struct vb2_queue vb2_queue;
    struct mutex queue_lock;
    struct mutex video_lock;
    struct media_device media_device;
};

struct hololink_buffer {
    struct vb2_v4l2_buffer vb;
    struct list_head list;
};

static int hololink_camera_v4l2_fh_open(struct file *filp)
{
    struct video_device * video_dev = video_devdata(filp);
    TRACE("filp=%p filp->private_data=%p video_dev=%p", filp, filp->private_data, video_dev);

    return v4l2_fh_open(filp);
}

static long hololink_camera_v4l2_fh_ioctl(struct file *filp, unsigned cmd, unsigned long arg)
{
    struct video_device * video_dev = video_devdata(filp);
    long ret;

    TRACE("filp=%p filp->private_data=%p video_dev=%p", filp, filp->private_data, video_dev);
    ret = video_ioctl2(filp, cmd, arg);
    TRACE("ret=%ld.", ret);
    return ret;
}

static const struct v4l2_file_operations hololink_camera_v4l2_fops = {
    .owner = THIS_MODULE,
    .open = hololink_camera_v4l2_fh_open,
    .release = v4l2_fh_release,
    .unlocked_ioctl = hololink_camera_v4l2_fh_ioctl,
};


static int hololink_vidioc_querycap(struct file *file, void *priv,
                struct v4l2_capability *cap)
{
    strscpy(cap->bus_info, "i2c", sizeof(cap->bus_info));
    strscpy(cap->driver, "hololink_camera", sizeof(cap->driver));
    strscpy(cap->card, "hololink-lite", sizeof(cap->card));
    return 0;
}

static int hololink_vidioc_enum_input(struct file *filp, void *priv, struct v4l2_input *i)
{
    if (i->index > 0) {
        return -EINVAL;
    }
    i->type = V4L2_INPUT_TYPE_CAMERA;
    strscpy(i->name, "Camera", sizeof(i->name));
    return 0;
}

static int hololink_vb2_ioctl_streamon(struct file *filp, void *priv, enum v4l2_buf_type i)
{
    struct video_device * video_dev = video_devdata(filp);
    struct hololink_camera_dev *hdev = container_of(video_dev, struct hololink_camera_dev, video_dev);
    int ret;

    TRACE("filp=%p priv=%p i=%d video_dev=%p hdev=%p.", filp, priv, (int) i, video_dev, hdev);

    ret = v4l2_subdev_call(hdev->v4l2_subdev, video, s_stream, true);
    TRACE("v4l2_subdev_call ret=%d.", ret);
    return ret;
}

static int hololink_vb2_ioctl_streamoff(struct file *filp, void *priv, enum v4l2_buf_type i)
{
    struct video_device * video_dev = video_devdata(filp);
    struct hololink_camera_dev *hdev = container_of(video_dev, struct hololink_camera_dev, video_dev);
    int ret;

    TRACE("filp=%p priv=%p i=%d video_dev=%p hdev=%p.", filp, priv, (int) i, video_dev, hdev);

    ret = v4l2_subdev_call(hdev->v4l2_subdev, video, s_stream, false);
    TRACE("v4l2_subdev_call ret=%d.", ret);
    return ret;
}


// Some callbacks that we're likely going to need to support
// are left here as comments-- when we have a geniune use and test
// for those, we'll implement these here.
static const struct v4l2_ioctl_ops hololink_camera_v4l2_ioctl_ops = {
    .vidioc_querycap = hololink_vidioc_querycap,
    .vidioc_enum_input = hololink_vidioc_enum_input,
    // .vidioc_g_input = hololink_vidioc_g_input,
    // .vidioc_s_input = hololink_vidioc_s_input,
    // .vidioc_enum_fmt_vid_cap = hololink_vidioc_enum_fmt_vid_cap,
    // .vidioc_g_fmt_vid_cap = hololink_vidioc_g_fmt_vid_cap,
    // .vidioc_s_fmt_vid_cap = hololink_vidioc_s_fmt_vid_cap,
    // .vidioc_try_fmt_vid_cap = hololink_vidioc_try_fmt_vid_cap,
    .vidioc_reqbufs = vb2_ioctl_reqbufs,
    .vidioc_create_bufs = vb2_ioctl_create_bufs,
    .vidioc_querybuf = vb2_ioctl_querybuf,
    .vidioc_qbuf = vb2_ioctl_qbuf,
    .vidioc_dqbuf = vb2_ioctl_dqbuf,
    .vidioc_expbuf = vb2_ioctl_expbuf,
    .vidioc_streamon = hololink_vb2_ioctl_streamon,
    .vidioc_streamoff = hololink_vb2_ioctl_streamoff,
    .vidioc_subscribe_event = v4l2_ctrl_subscribe_event,
    .vidioc_unsubscribe_event = v4l2_event_unsubscribe,
};

static int hololink_vb2_queue_setup(struct vb2_queue *vq,
    unsigned *nbuffers, unsigned *nplanes,
    unsigned int sizes[], struct device *alloc_devs[])
{
    TRACE("vq=%p nbuffers=%p nplanes=%p sizes=%p alloc_devs=%p",
        vq, nbuffers, nplanes, sizes, alloc_devs);
    return 0;
}

static int hololink_vb2_buffer_prepare(struct vb2_buffer *vb)
{
    TRACE("vb=%p", vb);
    return 0;
}

static void hololink_vb2_buffer_queue(struct vb2_buffer *vb)
{
    TRACE("vb=%p", vb);
}

static int hololink_vb2_start_streaming(struct vb2_queue *vq, unsigned int count)
{
    struct hololink_camera_dev *hdev = container_of(vq, struct hololink_camera_dev, vb2_queue);

    TRACE("vq=%p count=%u hdev=%p", vq, count, hdev);
    return 0;
}

static void hololink_vb2_stop_streaming(struct vb2_queue *vq)
{
    struct hololink_camera_dev *hdev = container_of(vq, struct hololink_camera_dev, vb2_queue);
    TRACE("vq=%p hdev=%p", vq, hdev);
}

static const struct vb2_ops hololink_vb2_ops = {
    .queue_setup = hololink_vb2_queue_setup,
    .buf_prepare = hololink_vb2_buffer_prepare,
    .buf_queue = hololink_vb2_buffer_queue,
    .start_streaming = hololink_vb2_start_streaming,
    .stop_streaming = hololink_vb2_stop_streaming,
    .wait_prepare = vb2_ops_wait_prepare,
    .wait_finish = vb2_ops_wait_finish,
};

static int hololink_v4l2_notify_bound(struct v4l2_async_notifier *notifier,
    struct v4l2_subdev *sd,
    struct v4l2_async_subdev *asd)
{
    struct hololink_camera_dev *hdev = container_of(notifier, struct hololink_camera_dev, v4l2_notifier);
    int ret;

    TRACE("notifier=%p sd=%p asd=%p hdev=%p.", notifier, sd, asd, hdev);
    TRACE("sd->devnode=%p sd->name=%s hdev->v4l2_dev=%p.", sd->devnode, sd->name, &hdev->v4l2_dev);

    // vb2_queue
    hdev->vb2_queue.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    hdev->vb2_queue.io_modes = VB2_MMAP | VB2_USERPTR | VB2_DMABUF;
    hdev->vb2_queue.drv_priv = hdev;
    hdev->vb2_queue.timestamp_flags = V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
    hdev->vb2_queue.buf_struct_size = sizeof(struct hololink_buffer);
    hdev->vb2_queue.dev = hdev->v4l2_dev.dev;
    hdev->vb2_queue.ops = &hololink_vb2_ops;
    hdev->vb2_queue.mem_ops = &vb2_dma_contig_memops;
    hdev->vb2_queue.lock = &hdev->queue_lock;

    ret = vb2_queue_init(&hdev->vb2_queue);
    TRACE("vb2_queue_init ret=%d", ret);

    // video_dev
    strscpy(hdev->video_dev.name, sd->name, sizeof(hdev->video_dev.name));
    hdev->video_dev.minor = -1;
    hdev->video_dev.fops = &hololink_camera_v4l2_fops;
    hdev->video_dev.ioctl_ops = &hololink_camera_v4l2_ioctl_ops;
    hdev->video_dev.release = video_device_release_empty;
    hdev->video_dev.device_caps = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING;
    hdev->video_dev.v4l2_dev = &(hdev->v4l2_dev);
    hdev->video_dev.ctrl_handler = sd->ctrl_handler;
    hdev->video_dev.queue = &hdev->vb2_queue;
    hdev->video_dev.lock = &hdev->video_lock;
    video_set_drvdata(&hdev->video_dev, hdev);
    ret = video_register_device(&hdev->video_dev, VFL_TYPE_VIDEO, -1);
    if (ret < 0) {
        printk(KERN_INFO "video_register_device ret=%d.\n", ret);
    }
    TRACE("hdev->video_dev=%p hdev->v4l2_dev.mdev=%p.", &hdev->video_dev, hdev->v4l2_dev.mdev);
    hdev->v4l2_subdev = sd;

    return 0;
}

static void hololink_v4l2_notify_unbind(struct v4l2_async_notifier *notifier,
    struct v4l2_subdev *sd,
    struct v4l2_async_subdev *asd)
{
    struct hololink_camera_dev *hdev = container_of(notifier, struct hololink_camera_dev, v4l2_notifier);

    TRACE("notifier=%p sd=%p asd=%p hdev=%p.", notifier, sd, asd, hdev);
    hdev->v4l2_subdev = NULL;
    video_unregister_device(&hdev->video_dev);
}

static const struct v4l2_async_notifier_operations hololink_v4l2_notifier_ops = {
    .bound = hololink_v4l2_notify_bound,
    .unbind = hololink_v4l2_notify_unbind,
};

static int hololink_camera_probe(struct platform_device *pdev)
{
    struct hololink_camera_dev *hdev;
    int ret;
    struct v4l2_async_subdev *asd;

    TRACE("pdev=%p dev=%p.", pdev, &pdev->dev);

    hdev = kzalloc(sizeof(*hdev), GFP_KERNEL);
    TRACE("hdev=%p.", hdev);
    if (hdev == NULL) {
        printk(KERN_INFO "hololink_camera_probe failed to allocate memory.\n");
        return -ENOMEM;
    }

    hdev->pdev = pdev;
    hdev->dev = &pdev->dev;
    platform_set_drvdata(pdev, hdev);

    hdev->media_device.dev = &pdev->dev;
    strscpy(hdev->media_device.model, "NVIDIA Hololink", sizeof(hdev->media_device.model));
    media_device_init(&hdev->media_device);

    ret = media_device_register(&hdev->media_device);
    TRACE("media_device_register ret=%d.", ret);
    if (ret < 0) {
        goto deallocate;
    }

    mutex_init(&hdev->queue_lock);
    mutex_init(&hdev->video_lock);

    hdev->v4l2_dev.mdev = &hdev->media_device;
    ret = v4l2_device_register(hdev->dev, &hdev->v4l2_dev);
    if (ret) {
        printk(KERN_INFO "hololink_camera_probe v4l2_device_register ret=%d.\n", ret);
        goto unregister_media_device;
    }

    v4l2_async_notifier_init(&hdev->v4l2_notifier);
    asd = v4l2_async_notifier_add_i2c_subdev(
        &hdev->v4l2_notifier,
        i2c_bus_number,
        i2c_device_address,
        struct v4l2_async_subdev);
    if (IS_ERR(asd)) {
        ret = PTR_ERR(asd);
        printk(KERN_INFO "hololink_camera_probe v4l2_async_notifier_add_i2c_subdev ret=%d.\n", ret);
        goto unregister;
    }

    hdev->v4l2_notifier.ops = &hololink_v4l2_notifier_ops;
    ret = v4l2_async_notifier_register(&hdev->v4l2_dev, &hdev->v4l2_notifier);
    if (ret) {
        printk(KERN_INFO "hololink_camera_probe v4l2_async_notifier_register ret=%d.\n", ret);
        goto notifier_cleanup;
    }

    TRACE("probed.");
    return 0;

notifier_cleanup:
    v4l2_async_notifier_cleanup(&hdev->v4l2_notifier);
unregister:
    v4l2_device_unregister(&hdev->v4l2_dev);
unregister_media_device:
    media_device_unregister(&hdev->media_device);
    media_device_cleanup(&hdev->media_device);
deallocate:
    kfree(hdev);
    return ret;
}

static int hololink_camera_remove(struct platform_device *pdev)
{
    struct hololink_camera_dev *hdev = platform_get_drvdata(pdev);

    TRACE("pdev=%p hdev=%p.", pdev, hdev);
    platform_set_drvdata(pdev, NULL);

    v4l2_async_notifier_unregister(&hdev->v4l2_notifier);
    v4l2_async_notifier_cleanup(&hdev->v4l2_notifier);
    media_device_unregister(&hdev->media_device);
    media_device_cleanup(&hdev->media_device);
    v4l2_device_unregister(&hdev->v4l2_dev);
    mutex_destroy(&hdev->video_lock);
    mutex_destroy(&hdev->queue_lock);

    kfree(hdev);
    return 0;
}

static void hololink_camera_pdev_release(struct device *dev)
{
    TRACE("dev=%p.", dev);
}

static struct platform_device hololink_camera_pdev = {
    .name = "hololink_camera",
    .dev = {
        .release = hololink_camera_pdev_release,
    },
};

static struct platform_driver hololink_camera_pdrv = {
    .driver = {
        .name = "hololink_camera",
    },
    .probe = hololink_camera_probe,
    .remove = hololink_camera_remove,
};

static int __init hololink_camera_video_init(void)
{
    int ret;

    ret = platform_device_register(&hololink_camera_pdev);
    TRACE("platform_device_register ret=%d.", ret);
    if (ret) {
        return ret;
    }

    ret = platform_driver_register(&hololink_camera_pdrv);
    TRACE("platform_driver_register ret=%d.", ret);
    if (ret) {
        platform_device_unregister(&hololink_camera_pdev);
    }

    return ret;
}
module_init(hololink_camera_video_init);

static void __exit hololink_camera_video_exit(void)
{
    TRACE("exiting.");
    platform_driver_unregister(&hololink_camera_pdrv);
    platform_device_unregister(&hololink_camera_pdev);
}
module_exit(hololink_camera_video_exit);

MODULE_AUTHOR("Patrick O'Grady <pogrady@nvidia.com>");
MODULE_DESCRIPTION("NVIDIA Hololink video driver");
MODULE_LICENSE("GPL v2");
