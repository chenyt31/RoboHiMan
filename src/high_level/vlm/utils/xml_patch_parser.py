import xml.etree.ElementTree as ET

def get_patch_center(x, y, w, h, k):
    cell_w = w / k
    cell_h = h / k
    center_x = int((x + 0.5) * cell_w)
    center_y = int((y + 0.5) * cell_h)
    return center_x, center_y

def extract_xml_fields(text: str, fields=("patch", "sub_task")):
    """
    从 XML 格式的文本中提取指定标签的内容和属性。
    
    Args:
        text (str): VLM 输出的 XML 字符串
        fields (tuple): 要解析的标签名，例如 ("patch", "sub_task")

    Returns:
        dict: {字段名: 对应解析结果}
    """
    results = {f: [] for f in fields}
    wrapped_text = f"<root>{text}</root>"

    try:
        root = ET.fromstring(wrapped_text)
    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
        return results

    for field in fields:
        for node in root.findall(field):
            if field == "patch":
                try:
                    x = node.attrib.get('x')
                    y = node.attrib.get('y')
                    if x is not None and y is not None:
                        results[field].append({
                            "patch": (int(x), int(y)),
                            "desc": node.text.strip() if node.text else None
                        })
                except Exception as e:
                    print("Patch Parse Error:", e)
                    results[field].append({
                        "patch": (0, 0),
                        "desc": "error"
                    })
            else:
                # 其他字段存储文本内容
                text_val = node.text.strip() if node.text else ""
                results[field].append(text_val)

    return results


def parse_vlm_output(text: str, patch_row_col=6, require_three=True, fields=("patch", "sub_task")):
    """
    解析 VLM 输出，提取 patch + 其他字段
    
    Args:
        text: VLM 输出
        patch_row_col: patch 网格数
        require_three: 是否强制 patch 数量为 3
        fields: 要解析的标签

    Returns:
        dict: 解析后的结果
    """
    decoded = extract_xml_fields(text, fields=fields)

    output = {}

    if "patch" in decoded:
        patch_result = [item["patch"] for item in decoded["patch"]]
        if require_three and len(patch_result) != 3:
            print("Warning: patch result is not 3, return default patch")
            patch_result = [(0, 0), (0, 0), (0, 0)]

        pixel_result = [
            get_patch_center(row, col, 224, 224, patch_row_col)
            for row, col in patch_result
        ]
        output["patch_result"] = patch_result
        output["pixel_result"] = pixel_result

    # 添加其他字段
    for f in fields:
        if f != "patch":
            output[f] = decoded.get(f, [])

    return output