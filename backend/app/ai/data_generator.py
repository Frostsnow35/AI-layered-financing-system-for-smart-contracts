"""
训练数据生成器
为AI风控模型生成模拟训练数据
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """训练数据生成器类"""
    
    def __init__(self):
        """初始化生成器"""
        self.random_seed = 42
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # 养老服务类型
        self.service_types = [
            "失能护理", "康复训练", "居家养老", "社区养老", 
            "医疗护理", "生活照料", "心理慰藉", "营养配餐"
        ]
        
        # 机构规模分类
        self.organization_sizes = ["小型", "中型", "大型"]
        
        # 地区分布
        self.regions = ["一线城市", "二线城市", "三线城市", "县级市"]
        
    def generate_synthetic_data(self, num_samples: int = 10000) -> pd.DataFrame:
        """生成合成训练数据"""
        try:
            logger.info(f"开始生成 {num_samples} 条合成训练数据...")
            
            data = []
            
            for i in range(num_samples):
                # 基础信息
                sample = self._generate_basic_info()
                
                # 财务信息
                sample.update(self._generate_financial_info(sample['organization_size']))
                
                # 运营信息
                sample.update(self._generate_operational_info(sample['service_type']))
                
                # 服务质量信息
                sample.update(self._generate_service_quality_info())
                
                # 风险标签
                sample['risk_label'] = self._generate_risk_label(sample)
                
                data.append(sample)
            
            df = pd.DataFrame(data)
            
            logger.info(f"合成数据生成完成，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"生成合成数据失败: {str(e)}")
            raise
    
    def _generate_basic_info(self) -> Dict[str, Any]:
        """生成基础信息"""
        return {
            'service_type': random.choice(self.service_types),
            'organization_size': random.choice(self.organization_sizes),
            'region': random.choice(self.regions),
            'establishment_years': random.randint(1, 20),
            'service_duration': random.randint(30, 365),  # 服务时长（天）
            'total_staff': random.randint(10, 200)
        }
    
    def _generate_financial_info(self, org_size: str) -> Dict[str, Any]:
        """生成财务信息"""
        # 根据机构规模确定基础金额
        size_multipliers = {"小型": 1, "中型": 3, "大型": 8}
        multiplier = size_multipliers[org_size]
        
        base_revenue = random.randint(50, 200) * multiplier  # 基础月收入（万元）
        
        # 生成财务数据
        monthly_revenue = base_revenue + random.uniform(-0.2, 0.3) * base_revenue
        total_assets = monthly_revenue * random.uniform(5, 15)
        total_liabilities = total_assets * random.uniform(0.1, 0.7)
        accounts_receivable = monthly_revenue * random.uniform(0.5, 2.0)
        operating_expenses = monthly_revenue * random.uniform(0.6, 0.9)
        
        # 政府补贴
        government_subsidy = monthly_revenue * random.uniform(0.05, 0.25)
        
        return {
            'monthly_revenue': monthly_revenue,
            'total_assets': total_assets,
            'total_liabilities': total_liabilities,
            'accounts_receivable': accounts_receivable,
            'operating_expenses': operating_expenses,
            'government_subsidy': government_subsidy,
            'government_subsidy_rate': government_subsidy / monthly_revenue
        }
    
    def _generate_operational_info(self, service_type: str) -> Dict[str, Any]:
        """生成运营信息"""
        # 根据服务类型调整参数
        service_params = {
            "失能护理": {"quality_base": 0.8, "satisfaction_base": 0.75, "staff_turnover_base": 0.15},
            "康复训练": {"quality_base": 0.85, "satisfaction_base": 0.8, "staff_turnover_base": 0.12},
            "居家养老": {"quality_base": 0.75, "satisfaction_base": 0.7, "staff_turnover_base": 0.18},
            "社区养老": {"quality_base": 0.8, "satisfaction_base": 0.75, "staff_turnover_base": 0.15},
            "医疗护理": {"quality_base": 0.9, "satisfaction_base": 0.85, "staff_turnover_base": 0.1},
            "生活照料": {"quality_base": 0.7, "satisfaction_base": 0.65, "staff_turnover_base": 0.2},
            "心理慰藉": {"quality_base": 0.75, "satisfaction_base": 0.7, "staff_turnover_base": 0.17},
            "营养配餐": {"quality_base": 0.8, "satisfaction_base": 0.75, "staff_turnover_base": 0.15}
        }
        
        params = service_params.get(service_type, service_params["生活照料"])
        
        return {
            'service_quality_score': min(100, max(0, params["quality_base"] * 100 + random.uniform(-15, 15))),
            'customer_satisfaction': min(100, max(0, params["satisfaction_base"] * 100 + random.uniform(-20, 20))),
            'staff_turnover_rate': max(0.01, min(0.5, params["staff_turnover_base"] + random.uniform(-0.05, 0.1))),
            'service_completion_rate': random.uniform(0.85, 0.98),
            'complaint_rate': random.uniform(0.001, 0.05),
            'average_service_hours': random.uniform(2, 8)
        }
    
    def _generate_service_quality_info(self) -> Dict[str, Any]:
        """生成服务质量信息"""
        return {
            'certification_level': random.choice(["无", "市级", "省级", "国家级"]),
            'staff_qualification_rate': random.uniform(0.6, 0.95),
            'equipment_completeness': random.uniform(0.7, 1.0),
            'safety_compliance_rate': random.uniform(0.8, 1.0),
            'training_frequency': random.uniform(0.1, 1.0)  # 每年培训次数
        }
    
    def _generate_risk_label(self, sample: Dict[str, Any]) -> int:
        """生成风险标签（0: 低风险, 1: 高风险）"""
        risk_score = 0
        
        # 财务风险因子
        debt_ratio = sample['total_liabilities'] / (sample['total_assets'] + 1)
        if debt_ratio > 0.6:
            risk_score += 0.3
        elif debt_ratio > 0.4:
            risk_score += 0.2
        
        # 运营风险因子
        if sample['staff_turnover_rate'] > 0.2:
            risk_score += 0.2
        elif sample['staff_turnover_rate'] > 0.15:
            risk_score += 0.1
        
        # 服务质量风险因子
        if sample['service_quality_score'] < 70:
            risk_score += 0.2
        elif sample['service_quality_score'] < 80:
            risk_score += 0.1
        
        if sample['customer_satisfaction'] < 70:
            risk_score += 0.15
        elif sample['customer_satisfaction'] < 80:
            risk_score += 0.05
        
        # 收入稳定性风险因子
        revenue_stability = random.uniform(0.1, 0.3)  # 模拟收入波动
        if revenue_stability > 0.25:
            risk_score += 0.2
        elif revenue_stability > 0.2:
            risk_score += 0.1
        
        # 政府补贴依赖风险因子
        if sample['government_subsidy_rate'] > 0.3:
            risk_score += 0.1
        
        # 随机因子
        risk_score += random.uniform(-0.1, 0.1)
        
        # 确保风险分数在0-1之间
        risk_score = max(0, min(1, risk_score))
        
        # 转换为二分类标签（阈值0.5）
        return 1 if risk_score > 0.5 else 0
    
    def generate_realistic_scenarios(self) -> List[Dict[str, Any]]:
        """生成现实场景数据"""
        scenarios = []
        
        # 场景1: 优质大型养老机构
        scenarios.append({
            'scenario': '优质大型养老机构',
            'service_type': '医疗护理',
            'organization_size': '大型',
            'monthly_revenue': 500,
            'total_assets': 8000,
            'total_liabilities': 2000,
            'service_quality_score': 95,
            'customer_satisfaction': 92,
            'staff_turnover_rate': 0.08,
            'government_subsidy_rate': 0.15,
            'expected_risk': 0  # 低风险
        })
        
        # 场景2: 中型稳定机构
        scenarios.append({
            'scenario': '中型稳定机构',
            'service_type': '失能护理',
            'organization_size': '中型',
            'monthly_revenue': 200,
            'total_assets': 3000,
            'total_liabilities': 1200,
            'service_quality_score': 85,
            'customer_satisfaction': 82,
            'staff_turnover_rate': 0.12,
            'government_subsidy_rate': 0.2,
            'expected_risk': 0  # 低风险
        })
        
        # 场景3: 小型新兴机构
        scenarios.append({
            'scenario': '小型新兴机构',
            'service_type': '居家养老',
            'organization_size': '小型',
            'monthly_revenue': 80,
            'total_assets': 800,
            'total_liabilities': 600,
            'service_quality_score': 78,
            'customer_satisfaction': 75,
            'staff_turnover_rate': 0.18,
            'government_subsidy_rate': 0.25,
            'expected_risk': 1  # 高风险
        })
        
        # 场景4: 高风险机构
        scenarios.append({
            'scenario': '高风险机构',
            'service_type': '生活照料',
            'organization_size': '小型',
            'monthly_revenue': 50,
            'total_assets': 400,
            'total_liabilities': 350,
            'service_quality_score': 65,
            'customer_satisfaction': 60,
            'staff_turnover_rate': 0.25,
            'government_subsidy_rate': 0.35,
            'expected_risk': 1  # 高风险
        })
        
        return scenarios
    
    def add_noise_to_data(self, df: pd.DataFrame, noise_level: float = 0.05) -> pd.DataFrame:
        """为数据添加噪声，提高真实性"""
        try:
            df_noisy = df.copy()
            
            # 为数值型列添加噪声
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col != 'risk_label':  # 不修改标签列
                    noise = np.random.normal(0, noise_level, len(df))
                    df_noisy[col] = df_noisy[col] * (1 + noise)
            
            logger.info(f"为数据添加了 {noise_level*100}% 的噪声")
            return df_noisy
            
        except Exception as e:
            logger.error(f"添加噪声失败: {str(e)}")
            raise
    
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """平衡数据集"""
        try:
            # 统计各类别数量
            class_counts = df['risk_label'].value_counts()
            logger.info(f"原始数据分布: {dict(class_counts)}")
            
            # 如果类别不平衡，进行过采样
            if len(class_counts) > 1:
                max_count = class_counts.max()
                min_count = class_counts.min()
                
                if max_count / min_count > 2:  # 如果比例超过2:1
                    # 对少数类进行过采样
                    minority_class = class_counts.idxmin()
                    minority_data = df[df['risk_label'] == minority_class]
                    
                    # 重复采样
                    oversample_size = max_count - min_count
                    oversample_data = minority_data.sample(
                        n=oversample_size, 
                        replace=True, 
                        random_state=42
                    )
                    
                    df_balanced = pd.concat([df, oversample_data], ignore_index=True)
                    
                    logger.info(f"数据平衡后分布: {dict(df_balanced['risk_label'].value_counts())}")
                    return df_balanced
            
            return df
            
        except Exception as e:
            logger.error(f"数据平衡失败: {str(e)}")
            raise
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """保存数据到文件"""
        try:
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"数据已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}")
            raise
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """从文件加载数据"""
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            logger.info(f"数据已从 {filepath} 加载，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            raise


# 创建全局数据生成器实例
data_generator = TrainingDataGenerator()
