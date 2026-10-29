import argparse
from models.CFSurv import network_20260414_CFSurv as cfsurv_net
from utils.config_utils import argparse_bool as str2bool, resolve_data_root_dir

def parse_args(model_name):
    parser = argparse.ArgumentParser(description="Configurations for Survival Analysis on TCGA Data.")

    parser.add_argument("model", type=str, default=model_name, help="Type of model (Default: CFSurv)")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default=None,
        help=(
            "WSI feature root. When omitted, use the first existing path from "
            "E:/Resnet50_Features_cyp and /home/user/Chensf/Dataset."
        ),
    )
    parser.add_argument("--extractor", type=str, default="resnet50", help="WSI feature extractor name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible experiment (default: 1)")
    parser.add_argument("--which_splits", type=str, default="5foldcv", help="Which splits folder to use in ./splits/ (Default: ./splits/5foldcv)")
    parser.add_argument("--sets", type=str, default="luad,ucec,blca,brca,gbmlgg", help='Which cancer type within ./splits/<which_dataset> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
    
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="Evaluate model on test set")
    parser.add_argument("--resume", type=str, default="", metavar="PATH", help="Path to latest checkpoint (default: none)")
    parser.add_argument("--fold", type=str, default="0,1,2,3,4", help='Which cancer type within ./splits/<which_dataset> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
    parser.add_argument("--modal", type=str, choices=["omic", "path", "multi"], default="multi", help="Specifies which modalities to use / collate function in dataloader.")
    parser.add_argument("--n_classes", type=int, default=4, help="Number of classes")

    parser.add_argument("--optimizer", type=str, choices=["Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead", "SGD"], default="AdamW")
    parser.add_argument("--scheduler", type=str, choices=["exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--num_epoch", type=int, default=30, help="Maximum number of epochs to train (default: 20)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size (Default: 1, due to varying bag sizes)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.00001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--loss", type=str, default="nllsurv,tri", help="survival loss function")
    parser.add_argument("--lambda_tri", type=float, default=1.0, help="Weight applied to the tri-factor loss")
    parser.add_argument("--nll_alpha", type=float, default=0.0, help="Uncensored-event weighting term used by the main survival NLL")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of batches accumulated before each optimizer step")
    parser.add_argument("--grad_clip_norm", type=float, default=0.0, help="Maximum gradient norm; 0 disables clipping")
    parser.add_argument("--optimizer_warmup_epochs", type=int, default=0, help="Linear optimizer warmup epochs before cosine scheduling")
    parser.add_argument("--experiment_name", type=str, default="", help="Optional custom suffix appended to result directory/file names.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--non_blocking", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_amp", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--tri_export_compression", choices=["none", "gzip"], default="none")
    parser.add_argument("--tri_export_chunksize", type=int, default=1000)
    parser.add_argument("--profile_flops", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--audit_wsi_features", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--token_audit", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--save_last_checkpoint", type=str2bool, nargs="?", const=True, default=True)

    model_specific_config = cfsurv_net.custom_config

    parser.set_defaults(**model_specific_config['base'])

    existing_option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    for k, v in model_specific_config['customized'].items():
        if f'--{k}' in existing_option_strings:
            continue
        vv = dict(v)
        vv['dest'] = k
        if vv.get("type", None) is bool:
            vv["type"] = str2bool
            vv.setdefault("nargs", "?")
            vv.setdefault("const", True)
        parser.add_argument('--' + k, **vv)

    args = parser.parse_args()
    args.loss = ','.join([tok.strip().lower() for tok in args.loss.replace('+', ',').split(',') if tok.strip()])
    try:
        args.data_root_dir = resolve_data_root_dir(args.data_root_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    return args
