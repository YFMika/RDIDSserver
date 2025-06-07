import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def convert_json_to_list_format(json_data, fields_to_convert):
    """
    将JSON数据中的指定字段从非列表格式转换为列表格式
    
    参数:
        json_data (dict): 解析后的JSON数据
        fields_to_convert (list): 需要转换的字段名列表
        
    返回:
        dict: 转换后的JSON数据
        bool: 是否发生了转换
    """
    converted = False
    
    for field in fields_to_convert:
        if field in json_data:
            # 如果字段存在且不是列表类型，则转换为列表
            if not isinstance(json_data[field], list):
                json_data[field] = [json_data[field]]
                converted = True
    
    return json_data, converted

def process_json_file(file_path, fields_to_convert, dry_run=False, overwrite=True, indent=2):
    """
    处理单个JSON文件，将指定字段转换为列表格式
    
    参数:
        file_path (str): JSON文件路径
        fields_to_convert (list): 需要转换的字段名列表
        dry_run (bool): 是否执行干跑模式（不修改文件）
        overwrite (bool): 是否覆盖原文件
        indent (int): JSON输出缩进空格数
        
    返回:
        int: 处理结果代码 (0=未修改, 1=已修改, 2=处理错误)
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        # 转换指定字段为列表格式
        converted_data, converted = convert_json_to_list_format(json_data, fields_to_convert)
        
        # 如果发生了转换且需要保存修改
        if converted and not dry_run:
            if overwrite:
                # 覆盖原文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(converted_data, f, indent=indent, ensure_ascii=False)
            else:
                # 保存为新文件（添加后缀）
                new_file_path = f"{file_path}.converted"
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    json.dump(converted_data, f, indent=indent, ensure_ascii=False)
            
            return 1  # 已修改
        return 0  # 未修改
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 2  # 处理错误

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='将JSON文件中的指定字段从非列表格式转换为列表格式'
    )
    
    # 添加命令行参数
    parser.add_argument('input', help='输入JSON文件或目录路径')
    parser.add_argument('--fields', nargs='+', default=['type', 'yanzhong'],
                        help='需要转换为列表格式的字段名，多个字段用空格分隔 (默认: type yanzhong)')
    parser.add_argument('--recursive', action='store_true',
                        help='递归处理子目录中的JSON文件')
    parser.add_argument('--dry-run', action='store_true',
                        help='执行干跑模式，不修改文件')
    parser.add_argument('--no-overwrite', action='store_true',
                        help='不覆盖原文件，保存为新文件（添加.converted后缀）')
    parser.add_argument('--indent', type=int, default=2,
                        help='JSON输出缩进空格数 (默认: 2)')
    parser.add_argument('--ext', default='.json',
                        help='要处理的文件扩展名 (默认: .json)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 确定处理模式（文件或目录）
    input_path = Path(args.input)
    
    # 初始化统计信息
    total_files = 0
    converted_files = 0
    error_files = 0
    
    # 处理单个文件
    if input_path.is_file():
        if input_path.suffix.lower() == args.ext.lower():
            total_files += 1
            result = process_json_file(
                str(input_path),
                args.fields,
                dry_run=args.dry_run,
                overwrite=not args.no_overwrite,
                indent=args.indent
            )
            
            if result == 1:
                converted_files += 1
                status = "✓ 已转换"
            elif result == 0:
                status = "○ 未修改"
            else:
                error_files += 1
                status = "✗ 处理错误"
                
            print(f"{status} {input_path}")
    
    # 处理目录
    elif input_path.is_dir():
        # 获取所有要处理的文件
        if args.recursive:
            files_to_process = list(input_path.rglob(f"*{args.ext}"))
        else:
            files_to_process = list(input_path.glob(f"*{args.ext}"))
        
        total_files = len(files_to_process)
        
        # 使用进度条处理文件
        print(f"开始处理 {total_files} 个JSON文件...")
        for file_path in tqdm(files_to_process, desc="处理进度"):
            result = process_json_file(
                str(file_path),
                args.fields,
                dry_run=args.dry_run,
                overwrite=not args.no_overwrite,
                indent=args.indent
            )
            
            if result == 1:
                converted_files += 1
            elif result == 2:
                error_files += 1
    
    else:
        print(f"错误: 路径不存在或不是有效文件/目录: {input_path}")
        return
    
    # 输出处理统计信息
    print("\n===== 处理统计 =====")
    print(f"总文件数: {total_files}")
    print(f"已转换文件: {converted_files}")
    print(f"未修改文件: {total_files - converted_files - error_files}")
    print(f"处理错误文件: {error_files}")
    
    if args.dry_run:
        print("\n⚠️ 注意: 由于使用了干跑模式，没有实际修改任何文件")

if __name__ == "__main__":
    main()    