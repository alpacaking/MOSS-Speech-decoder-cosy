import torch
import logging

def compare_two_state_dict(state_dict_A, state_dict_B):
    logging.info("########### compare_two_state_dict")
    # 排序关键字
    keys_A = sorted(state_dict_A.keys())
    keys_B = sorted(state_dict_B.keys())
    
    # 比较每个 key 的值
    missing_keys_A = []
    missing_keys_B = []
    
    # 找出在A中有但在B中没有的key
    for key in keys_A:
        if key not in state_dict_B:
            missing_keys_A.append(key)
        else:
            # 打印值的差异
            diff = torch.abs(state_dict_A[key] - state_dict_B[key]).mean()
            logging.info(f"Key: {key} | Difference: {diff}")
    
    # 找出在B中有但在A中没有的key
    for key in keys_B:
        if key not in state_dict_A:
            missing_keys_B.append(key)
    
    if missing_keys_A:
        logging.info(f"Keys missing in state_dict_B: {missing_keys_A}")
    if missing_keys_B:
        logging.info(f"Keys missing in state_dict_A: {missing_keys_B}")

def get_semantic_teachers_state_dict_from_students_state_dict(stu_state_dict):
    logging.info(f"####### get_semantic_teachers_state_dict_from_students_state_dict")
    # 新的字典，用来存储修改后的键值对
    teacher_state_dict = {}
    
    # 遍历stu_state_dict中的所有key-value对
    for key, value in stu_state_dict.items():
        # 检查key中是否包含"semantic_teacher."
        if "semantic_teacher." in key:
            # 去掉"semantic_teacher."前缀并作为新的key
            new_key = key.replace("semantic_teacher.", "")
            # 将新的key和原值添加到新的字典中
            teacher_state_dict[new_key] = value
    
    return teacher_state_dict

def show_grad(model):
    logging.info("########################### show grad")
    for name, param in model.named_parameters():
        if param.grad is not None:
            logging.info(f"{name} gradient norm: {param.grad.norm().item()}")
        else:
            logging.info(f"{name} gradient is None")
    