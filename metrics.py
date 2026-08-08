import os
import torch
import lpips
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from collections import defaultdict
from torchvision import transforms, models
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.image.fid import FrechetInceptionDistance


class FID:

    def __init__(
        self,
        batch_size: int = 50,
        dims: int = 2048,
        device: torch.device = None,
    ):
        """
        @param batch_size: 批处理大小
        @param device: cuda 或 cpu
        @param dims: Inception 特征维度，默认 2048 (Pool 3 层)
        """
        self.batch_size = batch_size
        self.dims = dims
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ]
        )

    def __call__(
        self,
        content_or_style_images: list[Image.Image],
        stylized_images: list[Image.Image],
    ) -> float:
        """
        计算两组 PIL Image 列表之间的 FID。

        Args:
            real_images (list): PIL.Image 列表（真实数据）
            fake_images (list): PIL.Image 列表（生成数据）
            batch_size (int): 批处理大小
            device (str): 运行设备 'cuda' 或 'cpu'
        """
        fid = FrechetInceptionDistance(feature=self.dims).to(self.device)

        def prepare_batches(img_list):
            tensors = torch.stack([self.transform(img) for img in img_list])
            return (tensors * 255).byte()

        real_tensors = prepare_batches(stylized_images)
        fake_tensors = prepare_batches(content_or_style_images)

        for i in range(0, len(real_tensors), self.batch_size):
            batch = real_tensors[i : i + self.batch_size].to(self.device)
            fid.update(batch, real=True)

        for i in range(0, len(fake_tensors), self.batch_size):
            batch = fake_tensors[i : i + self.batch_size].to(self.device)
            fid.update(batch, real=False)

        fid_score = fid.compute()
        return fid_score.item()


class LPIPS:

    def __init__(
        self,
        net: str = "vgg",
        return_dist=False,
        device: str = "cuda",
    ):
        """
        @param net: 'vgg', 'alex' 或 'squeeze' (默认使用 vgg)
        """
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.loss_fn.eval()

        self.return_dist = return_dist

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __call__(
        self,
        images1: list[Image.Image],
        images2: list[Image.Image],
    ) -> float:
        distances = []
        for img1, img2 in zip(images1, images2):
            img1 = self.transform(img1).unsqueeze(0).to(self.device)
            img2 = self.transform(img2).unsqueeze(0).to(self.device)
            with torch.no_grad():
                dist = self.loss_fn(img1, img2)
                distances.append(dist.item())
        avg_dist = sum(distances) / len(distances)
        if self.return_dist:
            return avg_dist, distances
        return avg_dist


class ArtFID:

    def __init__(
        self,
        batch_size: int = 50,
        fid_dims: int = 2048,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.fid_fn = FID(batch_size=self.batch_size, dims=fid_dims, device=self.device)
        self.lpips_fn = LPIPS(net="vgg", device=self.device)

    def __call__(
        self,
        content_images: list[Image.Image],
        style_images: list[Image.Image],
        stylized_images: list[Image.Image],
    ) -> float:
        fid_value = self.fid_fn(style_images, stylized_images)
        lpips_value = self.lpips_fn(content_images, stylized_images)
        art_fid_score = (1 + fid_value) * (1 + lpips_value)
        return art_fid_score


class CLIPImageScore:

    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([224, 224]),
            ]
        )

    def __call__(
        self,
        images1: list[Image.Image],
        images2: list[Image.Image],
    ) -> float:
        images1 = torch.stack([self.transform(img) for img in images1]).to(self.device)
        images2 = torch.stack([self.transform(img) for img in images2]).to(self.device)

        with torch.no_grad():
            feat_a = self.model.get_image_features(images1).pooler_output
            feat_b = self.model.get_image_features(images2).pooler_output

            feat_a = feat_a / feat_a.norm(p=2, dim=-1, keepdim=True)
            feat_b = feat_b / feat_b.norm(p=2, dim=-1, keepdim=True)

            similarity = (feat_a * feat_b).sum(dim=-1).mean()

        return similarity.cpu().item()


class StyleLoss:

    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.to(self.device).eval()

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.style_layers = {"0": "conv1_1", "5": "conv2_1", "10": "conv3_1", "19": "conv4_1", "28": "conv5_1"}

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def get_features(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.style_layers:
                features[self.style_layers[name]] = x
        return features

    def gram_matrix(self, x):
        """G = F * F^T"""
        (b, c, h, w) = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def __call__(
        self,
        style_images: list[Image.Image],
        stylized_images: list[Image.Image],
    ) -> float:
        style_images = torch.stack([self.transform(img) for img in style_images]).to(self.device)
        stylized_images = torch.stack([self.transform(img) for img in stylized_images]).to(self.device)

        gen_features = self.get_features(stylized_images)
        style_features = self.get_features(style_images)

        style_loss = 0
        for layer in self.style_layers.values():
            gen_gram = self.gram_matrix(gen_features[layer])
            style_gram = self.gram_matrix(style_features[layer])

            layer_loss = F.mse_loss(gen_gram, style_gram)
            style_loss += layer_loss

        return style_loss.item()


class PPL:

    def __init__(
        self,
        return_dist: bool = False,
        device: str = "cuda",
    ):
        self.loss_fn = lpips.LPIPS(net="vgg").to(device)
        self.device = device
        self.return_dist = return_dist

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __call__(
        self,
        images_t: list[Image.Image],
        images_next: list[Image.Image],
    ) -> float | tuple[float, list[float]]:
        """
        @param video_tensor: list of image (C, H, W) with T frames
        """
        T = min(len(images_t), len(images_next))
        total_ppl = 0.0

        all_dist = []
        for t in range(T):
            frame_t = self.transform(images_t[t]).to(self.device)
            frame_next = self.transform(images_next[t]).to(self.device)

            dist = self.loss_fn(frame_t, frame_next).item()
            all_dist.append(dist)
            total_ppl += dist

        if self.return_dist:
            return total_ppl, all_dist
        return total_ppl


class SPL:

    def __init__(
        self,
        return_dist: bool = False,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.return_dist = return_dist

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.to(self.device).eval()

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.style_layers = {"0": "conv1_1", "5": "conv2_1", "10": "conv3_1", "19": "conv4_1", "28": "conv5_1"}

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def get_features(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.style_layers:
                features[self.style_layers[name]] = x
        return features

    def gram_matrix(self, x):
        """计算 Gram 矩阵: G = F * F^T"""
        c, h, w = x.size()
        features = x.view(c, h * w)
        gram = features @ features.T
        return gram / (c * h * w)

    def __call__(
        self,
        images_t: list[Image.Image],
        images_next: list[Image.Image],
        style_image: Image.Image,
    ) -> float | tuple[float, list[float]]:
        """
        @param video_tensor: list of (C, H, W) with T frames
        """
        T = min(len(images_t), len(images_next))

        style_tensor = self.transform(style_image).to(self.device)
        style_features = self.get_features(style_tensor)

        total_dist = 0.0
        all_dist = []
        for t in range(T):
            frame_t = self.transform(images_t[t]).to(self.device)
            frame_next = self.transform(images_next[t]).to(self.device)
            gen_features_t = self.get_features(frame_t)
            gen_features_next = self.get_features(frame_next)

            style_loss_t = 0
            style_loss_next = 0
            for layer in self.style_layers.values():
                style_gram = self.gram_matrix(style_features[layer])

                gen_gram_t = self.gram_matrix(gen_features_t[layer])
                layer_loss_t = F.mse_loss(gen_gram_t, style_gram)

                gen_gram_next = self.gram_matrix(gen_features_next[layer])
                layer_loss_next = F.mse_loss(gen_gram_next, style_gram)

                style_loss_t += layer_loss_t
                style_loss_next += layer_loss_next

            dist = (style_loss_next - style_loss_t).abs().item()
            all_dist.append(dist)
            total_dist += dist

        if self.return_dist:
            return total_dist, all_dist
        return total_dist


class PDV:

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.ppl_fn = PPL(return_dist=True, device=device)

    def __call__(
        self,
        images_t: list[Image.Image],
        images_next: list[Image.Image],
    ) -> float:
        """
        @param video_tensor: list of (C, H, W) with T frames
        """
        _, distances = self.ppl_fn(images_t, images_next)
        dist_tensor = torch.tensor(distances)
        pdv = torch.var(dist_tensor, unbiased=True).item()
        return pdv


class SDV:

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.spl_fn = SPL(return_dist=True, device=device)

    def __call__(
        self,
        images_t: list[Image.Image],
        images_next: list[Image.Image],
        style_image: Image.Image,
    ) -> float:
        """
        @param video_tensor: list of (C, H, W) with T frames
        """
        _, distances = self.spl_fn(images_t, images_next, style_image)
        dist_tensor = torch.tensor(distances)
        sdv = torch.var(dist_tensor, unbiased=True).item()
        return sdv


class VideoStyleEvaluator:
    r"""
    Eval frames stylization performance
    """

    def __init__(self, result_dir, content_dir, style_dir, num_frames=25):
        self.result_dir = Path(result_dir)
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        self.num_frames = num_frames

        # 预加载路径映射
        self.data_map = self._scan_directories()

    def _scan_directories(self):
        """
        扫描文件夹并建立映射: frame_idx -> {content: [], style: [], stylized: []}
        """
        mapping = defaultdict(lambda: {"content": [], "style": [], "stylized": []})

        # 遍历子文件夹如 0001_0002
        subdirs = [d for d in self.result_dir.iterdir() if d.is_dir()]
        subdirs.sort()
        subdirs = subdirs[:50]

        for subdir in subdirs:
            try:
                cnt_id, sty_id = subdir.name.split("_")

                # 假设内容图和风格图的文件名规则，需根据实际情况微调
                # 例如内容图路径为 content_dir/0001.jpg
                cnt_path = self.content_dir / f"{cnt_id}.jpg"
                sty_path = self.style_dir / f"{sty_id}.jpg"

                for f_idx in range(self.num_frames):
                    frame_name = f"{f_idx:02d}.jpg"
                    stylized_path = subdir / frame_name

                    if stylized_path.exists():
                        mapping[f_idx]["content"].append(cnt_path)
                        mapping[f_idx]["style"].append(sty_path)
                        mapping[f_idx]["stylized"].append(stylized_path)
            except ValueError:
                continue  # 跳过格式不正确的文件夹
        return mapping

    def load_images(self, path_list):
        return [Image.open(p).convert("RGB") for p in path_list]

    def __call__(self, metrics_dict):
        """
        metrics_dict: 包含你定义的函数签名的字典
        """
        results = {}

        for f_idx in range(self.num_frames):
            print(f"--- Evaluating Frame {f_idx:02d} ---")
            frame_data = self.data_map[f_idx]
            if not frame_data["stylized"]:
                continue

            # 按需加载当前帧的所有图像（注意内存压力，如果数据量极大建议分批）
            cnt_imgs = self.load_images(frame_data["content"])
            sty_imgs = self.load_images(frame_data["style"])
            res_imgs = self.load_images(frame_data["stylized"])

            print(f"{len(cnt_imgs)=} | {len(sty_imgs)=} | {len(res_imgs)=}")

            frame_results = {}

            # 执行各项指标
            if "c-FID" in metrics_dict:
                frame_results["c-FID"] = metrics_dict["c-FID"](cnt_imgs, res_imgs)
            if "s-FID" in metrics_dict:
                frame_results["s-FID"] = metrics_dict["s-FID"](sty_imgs, res_imgs)
            if "LPIPS" in metrics_dict:
                frame_results["LPIPS"] = metrics_dict["LPIPS"](cnt_imgs, res_imgs)
            if "ArtFID" in metrics_dict:
                frame_results["ArtFID"] = metrics_dict["ArtFID"](cnt_imgs, sty_imgs, res_imgs)
            if "CLIPImageScore" in metrics_dict:
                frame_results["CLIPImageScore"] = metrics_dict["CLIPImageScore"](sty_imgs, res_imgs)
            if "StyleLoss" in metrics_dict:
                frame_results["StyleLoss"] = metrics_dict["StyleLoss"](sty_imgs, res_imgs)

            results[f_idx] = frame_results

            for k, v in frame_results.items():
                print(f"{k}: {v}")

        return results


class VideoSmoothnessEvaluator:
    r"""
    Eval video smoothness performance
    """

    def __init__(
        self,
        result_dir: str,
        style_dir: str,
        num_frames: int = 25,
        max_eval_num: int = 50,
        eval_start_idx: int = 0,
    ):
        self.result_dir = Path(result_dir)
        self.style_dir = Path(style_dir)
        self.num_frames = num_frames

        video_folders = [f for f in os.listdir(self.result_dir) if os.path.isdir(os.path.join(self.result_dir, f))]
        video_folders.sort()
        max_eval_num = min(max_eval_num, len(video_folders))
        self.video_folders = video_folders[eval_start_idx:max_eval_num]

    def __call__(self, metrics_dict):
        """
        metrics_dict: 包含你定义的函数签名的字典
        """
        results = {}

        for i, folder in enumerate(self.video_folders):
            print(f"--- Evaluating Video [{i+1}/{len(self.video_folders)}] {folder} ---")
            frames = [f for f in os.listdir(self.result_dir / folder) if Path(f).suffix in [".jpg", ".png"]]
            frames.sort()
            frames = frames[: self.num_frames]

            sty_id = folder.split("_")[1]
            sty_image = Image.open(self.style_dir / f"{sty_id}.jpg").convert("RGB")
            frame_images = [Image.open(self.result_dir / folder / p).convert("RGB") for p in frames]
            frames_t = frame_images[: self.num_frames - 1]
            frames_next = frame_images[1:]

            print(f"{len(frame_images)=}")

            video_result = {}
            if "PPL" in metrics_dict:
                video_result["PPL"] = metrics_dict["PPL"](frames_t, frames_next)
            if "SPL" in metrics_dict:
                video_result["SPL"] = metrics_dict["SPL"](frames_t, frames_next, sty_image)
            if "PDV" in metrics_dict:
                video_result["PDV"] = metrics_dict["PDV"](frames_t, frames_next)
            if "SDV" in metrics_dict:
                video_result["SDV"] = metrics_dict["SDV"](frames_t, frames_next, sty_image)

            results[folder] = video_result

            for k, v in video_result.items():
                print(f"{k}: {v}")

        return results


class ImageStyleEvaluator:

    def __init__(
        self,
        result_dir,
        content_dir,
        style_dir,
    ):
        self.result_dir = Path(result_dir)
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)

        # 自动扫描并对齐数据路径
        self.image_pairs = self._prepare_data()

    def _prepare_data(self):
        """
        解析 <cnt_id>_<sty_id>.jpg 并对齐 content/style 路径
        """
        pairs = []
        # 获取所有 jpg/png 文件
        img_files = list(self.result_dir.glob("*.jpg"))
        if not img_files:
            img_files = list(self.result_dir.glob("*.png"))

        print(f"Found {len(img_files)} stylized images. Aligning with datasets...")

        for img_path in img_files:
            name = img_path.stem  # 获取文件名（不含后缀）
            try:
                # 解析 ID
                cnt_id, sty_id = name.split("_")

                # 查找对应的原图 (支持多种后缀如 .jpg, .png)
                cnt_path = self._find_source(self.content_dir, cnt_id)
                sty_path = self._find_source(self.style_dir, sty_id)

                if cnt_path and sty_path:
                    pairs.append({"stylized": img_path, "content": cnt_path, "style": sty_path})
            except ValueError:
                print(f"Skipping invalid filename format: {img_path.name}")
                continue

        return pairs

    def _find_source(self, folder, img_id):
        for ext in [".jpg", ".png"]:
            p = folder / f"{img_id}{ext}"
            if p.exists():
                return p
        return None

    def __call__(self, metrics_dict):
        if not self.image_pairs:
            print("No valid image pairs found!")
            return {}

        print(f"Loading {len(self.image_pairs)} images into memory...")

        res_imgs = [Image.open(p["stylized"]).convert("RGB") for p in self.image_pairs]
        cnt_imgs = [Image.open(p["content"]).convert("RGB") for p in self.image_pairs]
        sty_imgs = [Image.open(p["style"]).convert("RGB") for p in self.image_pairs]

        results = {}

        for name, metric_fn in metrics_dict.items():
            print(f"Calculating {name}...")
            if "c-FID" in metrics_dict:
                results["c-FID"] = metrics_dict["c-FID"](cnt_imgs, res_imgs)
            if "s-FID" in metrics_dict:
                results["s-FID"] = metrics_dict["s-FID"](sty_imgs, res_imgs)
            if "LPIPS" in metrics_dict:
                results["LPIPS"] = metrics_dict["LPIPS"](cnt_imgs, res_imgs)
            if "ArtFID" in metrics_dict:
                results["ArtFID"] = metrics_dict["ArtFID"](cnt_imgs, sty_imgs, res_imgs)
            if "CLIPImageScore" in metrics_dict:
                results["CLIPImageScore"] = metrics_dict["CLIPImageScore"](cnt_imgs, res_imgs)
            if "StyleLoss" in metrics_dict:
                results["StyleLoss"] = metrics_dict["StyleLoss"](sty_imgs, res_imgs)

        return results
