# from ast import Store, parse
# import imp
# from multiprocessing import parent_process
#from multiprocessing.util import get_logger
from asyncio.log import logger
from base64 import encode
from calendar import prmonth
from cgi import test
from cgitb import text
from configparser import NoSectionError
from email.mime import image
from lib2to3.pgen2.tokenize import tokenize
from lib2to3.pytree import Node
from operator import iadd
import os 
import argparse
import math
import random
from sched import scheduler
from tkinter.ttk import Progressbar
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.utils.data import Dataset,DataLoader
import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from sd import sd
from PIL import ImageFile
from torch import autocast
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = get_logger(__name__)



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=False,
        help="path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_textencodermodel_name_or_path",
        type=str,
        default="openai/clip-vit-base-patch16",
        required=False,
        help="path to pretrained model or model identifier from huggingface.co/models.",

    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="pretrained tokenizer name or path if not the same as the model",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default='/hy-tmp/images',
        required=False,
        help="a folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default='*',
        help="a token to use as a placeholder for the concept"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="how many times to repeat the training data.",
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=5,
        help="number of save model epoch",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="a seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="all the images in the train/val dataset will be resized to this",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="whether to center crop images before resize",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="batch size(per device) for the training dataloder."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="batch size(per device) for the evaluation dataloder"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save_image_epoch",
        type=int,
        default=1,
        help="number of save image epoch",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="total number of training steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="number of updates steps to accumulate before performing a backward"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="the learning rate"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="the scheduler type to use"
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="number of steps for warmup in the lr_schdeler"
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="weight decay to use"
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="whether or not to push the model to the Hub"
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="the token for huggingface login "
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="the token to use to push to eht model hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether the model repository is private or not."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="for distributed trainging:local rank"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK",-1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.train_data_dir is None:
        raise ValueError("You must have train data directory")
    return args




class dreamboothdataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.image_paths = [os.path.join(self.data_root,file_path) for file_path in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set=="train":
            self._length = self.num_images*repeats
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length
    
    def __getitem__(self, i):
        example = {}
        if (i%(2*self.num_images))<self.num_images:
            
            image = Image.open(self.image_paths[i%self.num_images])
            if not image.mode == "RGB":
                image.convert("RGB")
          
            # placeholder_string = self.placeholder_token
            text = "A xtyui05 dog"  
            example["input_ids"] = self.tokenizer(
                text,padding="max_length", truncation=True, max_length=77 ,return_tensors="pt",
            ).input_ids[0]
            img = np.array(image).astype(np.uint8)

            if self.center_crop:
                crop = min(img.shape[0],img.shape[1])
                h,w, = (
                    img.shape[0],
                    img.shape[1],
                )
                img = img[(h-crop)//2:(h+crop)//2,(w-crop)//2:(w+crop)//2]
            image = Image.fromarray(img)
            image = image.resize((self.size,self.size),resample=self.interpolation)

            image = self.flip_transform(image)
            image = np.array(image).astype(np.uint8)
            image = (image/127.5-1.0).astype(np.float32)
            example["pixel_values"] = torch.from_numpy(image).permute(2,0,1)
        else:
            image = sd(prompt="A dog")
            text = "A dog"
            example["input_ids"] = self.tokenizer(
                text,padding="max_length", truncation=True, max_length=77 ,return_tensors="pt",
            ).input_ids[0]
            img = np.array(image).astype(np.uint8)
            if self.center_crop:
                crop = min(img.shape[0],img.shape[1])
                h,w, = (
                    img.shape[0],
                    img.shape[1],
                )
                img = img[(h-crop)//2:(h+crop)//2,(w-crop)//2:(w+crop)//2]
            image = Image.fromarray(img)
            image = image.resize((self.size,self.size),resample=self.interpolation)
            image = self.flip_transform(image)
            image = np.array(image).astype(np.uint8)
            image = (image/127.5-1.0).astype(np.float32)
            example["pixel_values"] = torch.from_numpy(image).permute(2,0,1)
        return example
def make_grid(images,rows,cols):
    w,h = images[0].size
    grid = Image.new('RGB',size=(cols*w,rows*h))
    for i, image in enumerate(images):
        grid.paste(image,box=(i%cols*w,i//cols*h))
    return grid
#evaluate the model
def evaluate(eval_batch_size,epoch,pipeline,prompt,out_dir):
    pipeline = pipeline.to("cuda")
    with autocast("cuda"):
        images = pipeline(batch_size = eval_batch_size,prompt = prompt,)["sample"]
    

    image_grid = make_grid(images,rows=2,cols=2)
    test_dir = os.path.join(out_dir,"samples")
    os.makedirs(test_dir,exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def freeze_params(params):
    for param in params:
        param.requires_grad = False

def main():
    args = parse_args()
    logging_dir = os.path.join(args.out_dir,args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )
    if args.seed is not None:
        set_seed(args.seed)
    #load the model
    YOUR_TOKEN="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"

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,subfolder="vae",use_auth_token=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder="tokenizer",use_auth_token=True)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="text_encoder",use_auth_token=True)
   
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",use_auth_token=True)
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    # text_encoder.resize_token_embeddings(len(tokenizer))
    #freeze parameters
    freeze_params(text_encoder.parameters())
    freeze_params(vae.parameters())
    optimizer_unet = torch.optim.AdamW(

        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,beta_end=0.012,beta_schedule="scaled_linear",num_train_timesteps=1000,tensor_format="pt"
    )

    train_dataset = dreamboothdataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True)


    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_unet,
        num_warmup_steps=args.lr_warmup_steps*args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps*args.gradient_accumulation_steps,
    )
    unet,optimizer,train_dataloader,lr_scheduler = accelerator.prepare(
        unet,optimizer_unet,train_dataloader,lr_scheduler
        )
    # to cuda
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    text_encoder.eval()
    unet.eval()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader)/args.gradient_accumulation_steps)
    args.num_epochs = math.ceil(args.max_train_steps/num_update_steps_per_epoch)


    if args.push_to_hub:
        repo = init_git_repo(args, at_init=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    #Train!
    total_batch_size = args.train_batch_size*accelerator.num_processes*args.gradient_accumulation_steps

    logger.info("*** Running training ***")
    logger.info(f" Num examples = {len(train_dataset)}")
    logger.info(f"Num epoches = {args.num_epochs}")
    logger.info(f"batch size = {args.train_batch_size}")
    logger.info(f" Total train batch size = {total_batch_size}")
    logger.info(f"Gradient Accmulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps = {args.max_train_steps}")
    
    progress_bar = tqdm(range(args.max_train_steps),disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0


    for epoch in range(args.num_epochs):
        # text_encoder.train()
        unet.train()
        # vae.train()
        for step,batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents*0.18215

                #Samle a noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                #Sample a timesteps for each image
                timesteps = torch.randint(0,noise_scheduler.num_train_timesteps,(bsz,)).long().to(latents.device)
                #add noise to the latents
                noisy_latents = noise_scheduler.add_noise(latents,noise,timesteps)
                
                
                #get the text_emedding
                encode_hidden_states = text_encoder(batch["input_ids"])[0]
                # print(encode_hidden_states.shape())
                # print(noisy_latents.shape())
                #predic the noise
                noise_pred = unet(noisy_latents,timesteps,encode_hidden_states)["sample"]
                loss = F.mse_loss(noise_pred,noise,reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)
                optimizer.step()
                # optimizer_text_encoder.step()
                # optimizer_unet.step()
                # optimizer_vae.step()
                lr_scheduler.step()
                # optimizer_text_encoder.zero_grad()
                # optimizer_vae.zero_grad()
                # optimizer_unet.zero_grad()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs,step=global_step)

            if global_step >= args.max_train_steps:
                break
        
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline(
                text_encoder = text_encoder,
                vae = vae,
                unet = accelerator.unwrap_model(unet),
                tokenizer = tokenizer,
                scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True),
                safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )
            if (epoch+1) % args.save_model_epochs ==0 or epoch == args.num_epochs-1:
                pipeline.save_pretrained(args.out_dir)
            if (epoch+1) % args.save_image_epoch ==0 or epoch == args.num_epochs-1:
                prompt="A xtyui05 dog"
                evaluate(args.eval_batch_size,epoch,pipeline,prompt,args.out_dir)
    accelerator.end_training()

if __name__ == "__main__":
    main()







    
    





