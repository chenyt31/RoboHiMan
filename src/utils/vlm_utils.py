import math
import xml.etree.ElementTree as ET
from point_renderer.rvt_renderer import RVTBoxRenderer as BoxRenderer
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Renderer:
    def __init__(self,
                 renderer_device='cuda:0',
                 img_size=224,
                 rend_three_views=True,
                 add_depth=False,
                 ):
        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            with_depth=add_depth,
        )

    def render(self, pc, img_feat):
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                img = [
                    self.renderer(
                        _pc,
                        _img_feat, # torch.cat((_pc, _img_feat), dim=-1),
                        fix_cam=True,
                        dyn_cam_info=None,
                    ).unsqueeze(0)
                    for _pc, _img_feat in zip(pc, img_feat)
                ]

        img = torch.cat(img, 0)
        img = img.permute(0, 1, 4, 2, 3)

        return img

# This is the resize function of Qwen2.5-VL
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)
    
    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))
    
    return [x1_new, y1_new, x2_new, y2_new]

def convert_to_qwen25vl_format_point(point, orig_height, orig_width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x, y = point
    x_new = round(x * scale_w)
    y_new = round(y * scale_h)
    
    x_new = max(0, min(x_new, new_width - 1))
    y_new = max(0, min(y_new, new_height - 1))
    
    return [x_new, y_new]

def convert_from_qwen25vl_format_point(point, orig_height, orig_width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height

    x_new, y_new = point
    x = round(x_new / scale_w)
    y = round(y_new / scale_h)

    x = max(0, min(x, orig_width - 1))
    y = max(0, min(y, orig_height - 1))

    return [x, y]

def decode_all_xml_points(text):
    """
    从多段 XML 字符串中提取所有 <points> 标签的坐标信息。
    """
    points_data = []
    try:
        wrapped_text = f"<root>{text}</root>"
        root = ET.fromstring(wrapped_text)
        for point_node in root.findall("points"):
            num_points = len(point_node.attrib) // 2
            single = []
            for i in range(num_points):
                x = point_node.attrib.get(f'x{i+1}')
                y = point_node.attrib.get(f'y{i+1}')
                if x is not None and y is not None:
                    single.append((int(x), int(y)))
            points_data.append({
                "points": single,
                "desc": point_node.text.strip() if point_node.text else None
            })
        return points_data
    except Exception as e:
        print("XML Parse Error:", e)
        return []

def decode_all_xml_patch(text):
    """
    从多段 XML 字符串中提取所有 <patch> 标签的坐标信息。
    """
    points_data = []
    wrapped_text = f"<root>{text}</root>"
    root = ET.fromstring(wrapped_text)
    for point_node in root.findall("patch"):
        try:
            x = point_node.attrib.get('x')
            y = point_node.attrib.get('y')
            if x is not None and y is not None:
                points_data.append({
                "patch": (int(x), int(y)),
                "desc": point_node.text.strip() if point_node.text else None
            })
        except Exception as e:
            print("XML Parse Error:", e)
            points_data.append({
                "patch": (0, 0),
                "desc": "error"
            })
    return points_data
    

def get_patch_index(x, y, w, h, k):
    cell_w = w // k
    cell_h = h // k
    x = min(x // cell_w, k - 1)
    y = min(y // cell_h, k - 1)
    return int(x), int(y)

def get_patch_center(x, y, w, h, k):
    cell_w = w / k
    cell_h = h / k
    center_x = int((x + 0.5) * cell_w)
    center_y = int((y + 0.5) * cell_h)
    return center_x, center_y

def visualize_grid_and_patches(img, k, title=None):
    """
    img: [C, H, W]，torch.Tensor 或 np.ndarray，值范围为 [0, 255] 或 [0.0, 1.0]
    k: patch 列行数（生成 k×k 网格）
    返回：带网格标注的图像，shape 与输入一致，类型保持一致
    """
    input_is_tensor = isinstance(img, torch.Tensor)
    orig_dtype = img.dtype if input_is_tensor else img.dtype

    if input_is_tensor:
        img = img.cpu().numpy()

    img = np.transpose(img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
    h, w = img.shape[1], img.shape[0]

    # figure 大小与原图像素一致，防止模糊
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # 填满整个画布
    ax.imshow(img.astype(np.uint8) if img.max() > 1 else img)
    
    cell_w = w // k
    cell_h = h // k

    # 画红色网格线
    for i in range(1, k):
        ax.axvline(x=i * cell_w, color='red', linewidth=1)
        ax.axhline(y=i * cell_h, color='red', linewidth=1)

    # 标注
    ax.text(cell_w // 2, cell_h // 2, "0", color='green', fontsize=12, ha='center', va='center', fontweight='bold')
    for j in range(1, k):
        ax.text(j * cell_w + cell_w // 2, cell_h // 2, f"{j}", color='green', fontsize=12, ha='center', va='center', fontweight='bold')
    for i in range(1, k):
        ax.text(cell_w // 2, i * cell_h + cell_h // 2, f"{i}", color='green', fontsize=12, ha='center', va='center', fontweight='bold')

    # 加标题
    if title is not None:
        ax.text(w // 2, h - 10, title, color='white', fontsize=16, ha='center', va='bottom', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0, edgecolor='none', pad=0))

    ax.axis('off')
    fig.canvas.draw()

    # 从 figure 中获取像素图像
    buf = np.asarray(fig.canvas.buffer_rgba())  # shape: (H, W, 4)
    out_img = buf[:, :, :3].copy()  # 丢弃 alpha 通道，得到 RGB 图像
    plt.close(fig)

    out_img = np.transpose(out_img, (2, 0, 1))  # [H, W, C] -> [C, H, W]

    if input_is_tensor:
        out_img = np.array(out_img, copy=True)  # Create a writable copy
        out_img = torch.from_numpy(out_img).to(dtype=orig_dtype)

    return out_img