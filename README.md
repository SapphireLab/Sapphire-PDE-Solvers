# PDESolvers

## 待办事项

- [ ] 重构: BVP IBVP 合并;
- [ ] 重构: BVP IBVP 定义域和方程解耦;
- [ ] 问题: ELM 求解高对比跳跃系数时效果很差;
- [ ] 问题: ELM 求解真解形式如 sin(kx)sin(ky)sin(kz) 时效果很差;
- [ ] 疑问: PoU 实现需要再进行检查;
- [ ] 优化: PoU 分量图调整上下, 与复合图对应;
- [ ] BUG: PoU_C (1×1) 时与 PoU_Base 存在差异;
- [ ] BUG: PINN 类型的训练信息打印方式
- [ ] BUG: 'align' 权重分配计算时 log_10(1) 判断为-1 需要修复;
- [x] 需求: 高维问题更换真解 (ELM, GELM, RFM, RBFELM) 50%;
- [x] 需求: 圆形界面问题调优 (ELM, GELM, RFM, RBFELM) 10%;
- [ ] BUG: 3×3 PoU 出现 NaN 结果;
- [ ] 错误: IFVE07, IFVE08 界面与论文不对应;
- [ ] 实现: PMBM10 多界面问题;
- [ ] 实现: 论文对应算子替换;
- [ ] 优化: 五角星界面简化;
- [ ] 优化: 界面问题配置文件完善定义域等细节;
- [ ] 实现: 用~~界面问题~~边值问题的结构实现奇性问题;
  用界面问题进行边界采样过于繁杂, 且无法均匀采样, 因此放弃;
- [ ] 优化: 给保存器 Saver 增加模型保存;
- [ ] 疑问: 旋度算子是否需要二次创建计算图;
- [ ] 文档: 完善界面问题文档;

## 完成事项

- [x] 结果: 提交抛物界面问题结果 20240119
- [x] 问题: 三维情形的平均曲率出现问题?;
- [x] 复现: RFM 抛物界面问题算例;
- [x] 实现: PMBM02 03 参数方程类型界面问题;
- [x] 结果: 提交抛物界面问题初步报告 20231225
- [x] 结果: 提交抛物界面问题初步报告 20231218
- [x] 重构: 界面问题绘图与结果保存;
- [x] 问题: IFEMnew 第三个问题计算失败 -> 真解定义的次数不合理;
- [x] 重构: 重构了界面问题的绘图逻辑;
- [x] 修改: trainIP 和 IPScript 合并, 删除冗余部分;
- [x] 修复: ELM 适配抛物问题;
- [x] 实现: 新增 IFEM 算例, SGFEM 算例;
- [x] 重构: IP_Config 完全拆分到复现文件中;
- [x] 实现：奇异震荡问题，Δu项系数为1.0时可得1e-10, 在10-8时仅有1e-1，考虑 X-TFC;
- [x] 实现: LRNN 移动界面问题;
- [x] 实现: LRNN 抛物界面问题 (时间切片绘图);
- [x] 重构: 解耦界面问题的定义域与方程;
- [x] 实现: 初边值问题;
- [x] BUG: 现有的 PoU 无法支持超过 32 维的问题 (np.mesh 限制子区域划分);
解决方案: 只对需要划分的维度进行 meshgrid, 其他维度复制.
- [x] 复现: ShallowPINN 用于椭球曲面问题 demo, 运算速度缓慢约 1.5 iter/s
- [x] 实现：Hessian 矩阵算子
- [x] 实现：曲面问题 (J.CMA.2023.116486_A_Shallow_PINN_for_Solving_PDEs_on_Static_&Evolving_Surfaces.pdf) 及 RFM 求解器
  平均曲率图验证通过.
- [x] 疑问：曲面问题算子应当拆成三部分, 拆分效果稍好但仍不佳，趋势相同但值偏移很大.
- [x] BUG: 'align' 权重分配缺少记录数据;
- [x] 修改: 将 ELM 模型重写为全连接层, 与 PoU 解耦, 以支持高维问题.
- [x] 优化: 配置列表修改为配置字典, 以便通过问题编号对应配置;
- [x] 新增: 高维 BVP 配置文件 (BVPConfig.py);
- [x] 新增: 高维超平面界面问题 (IP.py > class HyperplaneInterface);
- [x] 新增: 论文 LRNN 算例配置;
- [x] 优化: InterfaceProblem 支持非分量形式函数输入 (interface_func_xi, u_inner_xi, u_outer_xi)
即自变量为整个样本点 X, 而非分量 x,y,z, 为了兼容高维输入.

## 项目结构

Problems:
- 基本算子 _Operator.py
- 边值问题 [BVP.py](Problems/BVP.py)
  - \<class\> BoundaryValueProblem: 问题定义;
    - 基本属性: 区域下界 low, 区域上界 high, 维|数 dim, 设备 device
    - 基本函数: 数组转张量, 真解, PDE 算子, PDE 条件, 边界条件. 
  - \<class\> BVP_Sampler: BVP 采样器;
    - 基本属性: 区域下界, 区域上界, 维|数,
    - 基本函数: 边界生成, 区域采样, 边界采样
  - \<class\> BVP_Plotter: BVP 绘图器;
    - 基本属性: 区域下界, 区域上界, 维|数,
    - 基本函数: #TODO 采样点显示, 结果二维|对比图 (高维|情形取切片);
  - \<class\> BVP_Solver: BVP 求解器基本框架;
    - 基本属性: 问题实例, 模型实例, 采样器实例, 绘图器实例;
    - 基本函数: 训练数据生成, 训练, 测试 
- 时空问题 IBVP.py
- 界面问题 IP.py
- 奇性问题 SP.py

Models:
- 基本神经网络 _Network.py
- 极端学习机 ELM.py
- 随机特征方法 RFM.py
- 广义极端学习机 GELM.py (RFM + PoU_C)
- 物理信息神经网络 PINN.py

Utils:
- BaseFunc.py
- SampleFunc.py
- PlotFunc.py

Results: 用于存储相应的实验结果, 方便复现.
