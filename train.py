#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import os.path as osp
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
import random
import uuid
import cv2
sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams, ModelHiddenParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, l1_loss_weighted, ssim, tv_3d_loss, normalized_cross_correlation
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice
import torch.nn.functional as F



def crop(orig_image, k, j, ul):
    x = orig_image.shape[1]
    y = orig_image.shape[2]
    # Create a normalized grid for sampling
    # Normalize the coordinates to range [-1, 1] for grid_sample
    grid_x = torch.linspace(ul[0] / (x - 1) * 2 - 1, (ul[0] + k - 1) / (x - 1) * 2 - 1, k)
    grid_y = torch.linspace(ul[1] / (y - 1) * 2 - 1, (ul[1] + j - 1) / (y - 1) * 2 - 1, j)
    grid_y, grid_x = torch.meshgrid(grid_x, grid_y, indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).to('cuda')  # Shape [1, k, j, 2]

    # Use grid_sample to sample the sub-region with interpolation
    cropped_image = F.grid_sample(orig_image.unsqueeze(0), grid, mode='bilinear', align_corners=True)

    # Remove the extra batch dimension
    cropped_image = cropped_image.squeeze(0)
    return cropped_image


def training(
    dataset: ModelParams,
    hp: ModelHiddenParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    first_iter = 0
    first_iter = 30000

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min and dataset.scale_max:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x, s, t: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
        s, #stage
        t, #time
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound, hp)
    initialize_gaussian(gaussians, dataset)
    scene.gaussians = gaussians

    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)

    first_iter += 1
    vol_gt = scene.vol_gt["mean"]
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        if iteration < 8000:
            stage = 'coarse'
        else:
            stage = 'fine'
        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, pipe, stage)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        # crop the the detector region
        gt_image = viewpoint_cam.original_image.cuda()
        image = crop(image, gt_image.shape[1], gt_image.shape[2], scanner_cfg["crop_ul"])
        # Compute loss
        #
        if True and iteration % 50 == 0:
            # Ensure images are in [0, 255] range for OpenCV (if not already)
            gt_image_np = gt_image.squeeze().cpu().detach().numpy()
            pred_img_np = image.squeeze().cpu().detach().numpy()

            # 1. Combine the two arrays to calculate the global min and max
            combined_min = min(gt_image_np.min(), pred_img_np.min())
            combined_max = max(gt_image_np.max(), pred_img_np.max())

            # 2. Normalize both arrays using the combined min and max
            gt_image_np = (gt_image_np - combined_min) / (combined_max - combined_min)
            pred_img_np = (pred_img_np - combined_min) / (combined_max - combined_min)

            # 3. Scale to [0, 255] and convert to uint8
            gt_image_np = (gt_image_np * 255).astype(np.uint8)
            pred_img_np = (pred_img_np * 255).astype(np.uint8)

            # 4. Display ground truth and prediction side by side
            combined_image = np.hstack((gt_image_np, pred_img_np))  # Stack images horizontally
            cv2.imshow('Ground Truth (Left) vs Prediction (Right)', combined_image)
            # Wait briefly to allow the image to update
            cv2.waitKey(1)  # 1 ms delay

        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        # render_loss = 1.0 - normalized_cross_correlation(image, gt_image)
        loss["render_l1"] = render_loss
        loss["total"] += loss["render_l1"]

        if opt.lambda_dssim > 0:
            # loss_dssim = 0
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim

        if stage == "fine" and hp.time_smoothness_weight != 0:
            # tv_loss = 0
            hexplane_tv_loss = gaussians.compute_regulation(hp.time_smoothness_weight, hp.l1_time_planes, hp.plane_tv_weight)
            loss["hexplane_tv"] = hexplane_tv_loss
            loss["total"] = loss["total"] + hexplane_tv_loss
        else:
            loss["hexplane_tv"] = torch.tensor(0.0)
        # 3D TV loss of CT volume (not used)
        if False:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
                stage,
            )["vol"]
            # np.save('vol_pred.npy', vol_pred.detach().cpu().numpy())
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["ct_tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv
        use_cgls_to_regularize = False
        loss["ct_vol_loss"] = torch.tensor(0.0)
        if use_cgls_to_regularize and iteration%30==0:
            vol_pred = queryfunc(scene.gaussians, stage, None)["vol"]
            vol_loss = l1_loss(vol_pred, vol_gt)
            loss["ct_vol_loss"] = vol_loss
            loss["total"] = loss["total"] + 0.005 * vol_loss


        if iteration % 10 == 0:
            loss_total = loss["total"]
            loss_render_l1 = loss["render_l1"]
            loss_ct_vol = loss["ct_vol_loss"]
            tqdm.write(
                f"[ITER {iteration}] TotalLoss: {loss_total}, ct_vol_loss: {loss_ct_vol}, RenderLoss: {loss_render_l1}, dssim: {loss_dssim}"
            )

        loss["total"].backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y, stage: render(x, y, pipe, stage),
                queryfunc,
                stage,
                scanner_cfg
            )


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
    stage,
    scanner_cfg,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                        stage,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    image = crop(image, gt_image.shape[1], gt_image.shape[2], scanner_cfg["crop_ul"])

                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=None,
                                    vmax=None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance
        key_t = 'mean'
        if stage == 'coarse':
            vol_pred = queryFunc(scene.gaussians, stage, None)["vol"]
        elif stage == 'fine':
            tt = random.randint(0, 9)
            key_t = str(tt * 10).zfill(2)
            vol_pred = queryFunc(scene.gaussians, stage, tt/10.)["vol"]
        else:
            RuntimeError("stage is coarse or fine only!")

        # vol_gt = scene.vol_gt[key_t]
        vol_gt = scene.vol_gt["mean"]
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        eval_dict = {
            f"{key_t}_psnr_3d": psnr_3d,
            f"{key_t}_ssim_3d": ssim_3d,
            f"{key_t}_ssim_3d_x": ssim_3d_axis[0],
            f"{key_t}_ssim_3d_y": ssim_3d_axis[1],
            f"{key_t}_ssim_3d_z": ssim_3d_axis[2],
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        # np.save(osp.join(eval_save_path, "vol_pred.npy"), vol_gt.cpu().numpy())
        if tb_writer:
            image_show_3d = np.concatenate(
                [
                    show_two_slice(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"slice {i} gt",
                        f"slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vol_gt[..., i].max(),
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
    )

    # All done
    print("Training complete.")
