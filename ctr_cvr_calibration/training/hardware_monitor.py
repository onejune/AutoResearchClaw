"""
硬件资源监控与动态选择模块

功能:
1. 实时监控 GPU/CPU 利用率和内存
2. 根据负载动态选择训练设备
3. 优先 GPU，GPU 不可用时自动降级到 CPU
"""

import os
import subprocess
import torch
import psutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DeviceType(Enum):
    """设备类型"""
    GPU = "cuda"
    CPU = "cpu"


@dataclass
class DeviceInfo:
    """设备信息"""
    device_type: DeviceType
    device_id: int = 0
    name: str = ""
    utilization: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_available_mb: float = 0.0
    compute_capability: str = ""


class HardwareMonitor:
    """硬件资源监控器"""
    
    def __init__(self):
        self.gpus = self._detect_gpus()
        self.cpu = self._detect_cpu()
    
    def _detect_gpus(self) -> List[DeviceInfo]:
        """检测所有可用 GPU"""
        gpus = []
        
        if not torch.cuda.is_available():
            return gpus
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            utilization = self._get_gpu_utilization(i)
            memory_used = self._get_gpu_memory_used(i)
            
            gpus.append(DeviceInfo(
                device_type=DeviceType.GPU,
                device_id=i,
                name=props.name,
                utilization=utilization,
                memory_used_mb=memory_used,
                memory_total_mb=props.total_memory / (1024**2),
                memory_available_mb=(props.total_memory / (1024**2)) - memory_used,
                compute_capability=f"{props.major}.{props.minor}"
            ))
        
        # 按利用率升序排序（优先选择空闲的）
        return sorted(gpus, key=lambda x: x.utilization)
    
    def _detect_cpu(self) -> DeviceInfo:
        """检测 CPU 信息"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return DeviceInfo(
            device_type=DeviceType.CPU,
            device_id=0,
            name=f"CPU ({psutil.cpu_count(logical=False)} cores)",
            utilization=cpu_percent,
            memory_used_mb=memory.used / (1024**2),
            memory_total_mb=memory.total / (1024**2),
            memory_available_mb=memory.available / (1024**2)
        )
    
    def _get_gpu_utilization(self, gpu_id: int) -> float:
        """获取 GPU 利用率 (%)"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', 
                 '--format=csv,nounits', '-i', str(gpu_id)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                util_str = result.stdout.strip().split('\n')[-1].strip()
                return float(util_str)
        except Exception:
            pass
        return 0.0
    
    def _get_gpu_memory_used(self, gpu_id: int) -> float:
        """获取 GPU 已用显存 (MB)"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', 
                 '--format=csv,nounits', '-i', str(gpu_id)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                mem_str = result.stdout.strip().split('\n')[-1].strip()
                return float(mem_str)
        except Exception:
            pass
        return 0.0
    
    def select_device(
        self,
        min_memory_mb: float = 1024,
        max_utilization: float = 90.0,
        prefer_gpu: bool = True
    ) -> Tuple[torch.device, DeviceInfo]:
        """
        选择最优设备
        
        Args:
            min_memory_mb: 最小内存需求 (MB)
            max_utilization: 最大利用率阈值
            prefer_gpu: 是否优先 GPU
        
        Returns:
            (device, device_info): 设备对象和设备信息
        """
        print("\n" + "="*60)
        print("硬件资源监控")
        print("="*60)
        
        # 打印 CPU 信息
        self._refresh_cpu()
        print(f"\n💻 CPU:")
        print(f"   名称: {self.cpu.name}")
        print(f"   利用率: {self.cpu.utilization:.1f}%")
        print(f"   内存: {self.cpu.memory_used_mb:.0f}MB / {self.cpu.memory_total_mb:.0f}MB")
        print(f"   可用: {self.cpu.memory_available_mb:.0f}MB")
        
        # 打印 GPU 信息
        if self.gpus:
            print(f"\n🎮 GPU ({len(self.gpus)} 个):")
            for gpu in self.gpus:
                status = "✅ 空闲" if gpu.utilization < max_utilization else "⚡ 繁忙"
                print(f"   GPU {gpu.device_id}: {gpu.name}")
                print(f"      利用率: {gpu.utilization:.1f}% [{status}]")
                print(f"      显存: {gpu.memory_used_mb:.0f}MB / {gpu.memory_total_mb:.0f}MB")
                print(f"      可用: {gpu.memory_available_mb:.0f}MB")
        else:
            print(f"\n🎮 GPU: 无可用 GPU")
        
        # 选择设备
        selected_device = None
        reason = ""
        
        if prefer_gpu and self.gpus:
            # 优先 GPU：找第一个满足条件的
            for gpu in self.gpus:
                if (gpu.utilization < max_utilization and 
                    gpu.memory_available_mb >= min_memory_mb):
                    selected_device = gpu
                    reason = f"GPU {gpu.device_id} 空闲且显存充足"
                    break
            
            # 如果没有空闲 GPU，选择利用率最低的
            if selected_device is None and self.gpus:
                selected_device = self.gpus[0]  # 已按利用率排序
                reason = f"GPU {selected_device.device_id} 利用率最低"
        
        # 如果没有 GPU 或 GPU 不满足条件，使用 CPU
        if selected_device is None:
            selected_device = self.cpu
            reason = "使用 CPU (无可用 GPU 或 GPU 不满足条件)"
        
        # 返回设备
        if selected_device.device_type == DeviceType.GPU:
            device = torch.device(f"cuda:{selected_device.device_id}")
        else:
            device = torch.device("cpu")
        
        print(f"\n✅ 选中设备: {selected_device.name}")
        print(f"   类型: {selected_device.device_type.value}")
        print(f"   原因: {reason}")
        print("="*60 + "\n")
        
        return device, selected_device
    
    def _refresh_cpu(self):
        """刷新 CPU 信息"""
        self.cpu.utilization = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        self.cpu.memory_used_mb = memory.used / (1024**2)
        self.cpu.memory_available_mb = memory.available / (1024**2)
    
    def refresh(self):
        """刷新所有设备信息"""
        self.gpus = self._detect_gpus()
        self._refresh_cpu()
    
    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            'gpus': [
                {
                    'id': gpu.device_id,
                    'name': gpu.name,
                    'utilization': gpu.utilization,
                    'memory_used_mb': gpu.memory_used_mb,
                    'memory_available_mb': gpu.memory_available_mb
                }
                for gpu in self.gpus
            ],
            'cpu': {
                'utilization': self.cpu.utilization,
                'memory_available_mb': self.cpu.memory_available_mb
            }
        }


def select_training_device(
    min_memory_mb: float = 1024,
    max_utilization: float = 90.0
) -> torch.device:
    """
    便捷函数：选择训练设备
    
    Args:
        min_memory_mb: 最小内存需求 (MB)
        max_utilization: 最大利用率阈值
    
    Returns:
        torch.device: 选中设备
    """
    monitor = HardwareMonitor()
    device, _ = monitor.select_device(
        min_memory_mb=min_memory_mb,
        max_utilization=max_utilization,
        prefer_gpu=True
    )
    return device


if __name__ == '__main__':
    print("=== 硬件资源监控测试 ===\n")
    
    # 测试设备选择
    monitor = HardwareMonitor()
    device, info = monitor.select_device(min_memory_mb=4096)
    
    print(f"\n选中设备: {device}")
    print(f"设备信息: {info}")
    
    # 测试状态获取
    status = monitor.get_status()
    print(f"\n状态: {status}")
