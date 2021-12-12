import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference, do_inference_uda
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    if cfg.MODEL.UDA_STAGE == 'UDA':
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, train_loader1, train_loader2, img_num1, img_num2, s_dataset, t_dataset = make_dataloader(cfg)
    else:
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    
    # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param_finetune(cfg.TEST.WEIGHT)
    if cfg.MODEL.UDA_STAGE == 'UDA':
        do_inference_uda(cfg,
                 model,
                 val_loader,
                 num_query)
    else:
        do_inference(cfg,
                    model,
                    val_loader,
                    num_query)
