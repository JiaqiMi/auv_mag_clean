# AUV 磁洁净数字试验台

这是一个本地运行、可交互的“台架标定—航行前馈—铯光泵观测”模型在环仿真。它用于在实物研制前发现方案中的量级矛盾、模型泛化不足、线圈饱和和目标信号误伤风险。

Web 面板可直接配置推进器、DC/DC、作动器和电子负载相对铯光泵敏感体积的三维坐标。运行后，三维动态面板会同步显示各等效源的工况状态、在铯光泵位置产生的场矢量、保护线圈反向场以及补偿后的残差。默认进入“高机动越限验证”：启动突增、母线阶跃、转向、推进反转和紧急加速会使无补偿全梯度超过 G-882 的 10000 nT/m 容限，界面同时显示补偿前后峰值、越限时长和合规读数占比。

为把高机动场残差压到 1 nT 以下，当前流程增加了“航前独立迁移标定”和“驱动一阶逆预加重”：前者只使用可拆卸阵列与独立航前激励，校正台架模型到当前艇体/线圈的传递变化；后者只根据当前与上一拍前馈需求补偿已辨识的驱动滞后。两者在入水前固化，均不读取航行铯光泵或外部目标。默认仿真还保留航前标定后的 0.03% 源/线圈短时剩余漂移和 5% 驱动时间常数辨识误差，避免假设参数完全不变或逆模型完全准确。

当前四路工况仍是归一化等效激励；只有推进器通道通过“1.0 = 20 A参考电流”给出等效电流显示。自磁强度倍率以及线圈能力倍率是可审计的压力试验参数，不是实物测量结果，后续必须用实测电流、转速、PWM、回流路径、等效磁矩和线圈标定矩阵替换。

## 立即运行

在本目录打开 PowerShell：

```powershell
python run.py
```

浏览器会打开 `http://127.0.0.1:8050`。若不希望自动打开浏览器：

```powershell
python run.py --no-browser
```

命令行单次运行及保存数据：

```powershell
python simulate_cli.py --config configs/default.json --save
```

严格稳健性矩阵：

```powershell
python robustness_cli.py --seeds 10
```

复现本轮源模型升级的旧—新模型对照、10随机种子和源漂移压力测试：

```powershell
python validate_upgrade.py --seeds 10
```

复现界面四档工况预设的40案例验证：

```powershell
python validate_presets.py --seeds 10
```

单独复现高机动 `<1 nT` 目标的10随机种子、消融和梯度安全检查：

```powershell
python validate_field_goal.py --seeds 10
```

## 代码框架

```text
auv_magnetic_sim/
├─ magclean_sim/
│  ├─ config.py          # 所有可调参数和工程单位
│  ├─ math3d.py          # 偶极场、坐标旋转、场—梯度重建
│  ├─ sources.py         # AUV 多源自磁干扰正演
│  ├─ coils.py           # 真实线圈、温漂、驱动动态
│  ├─ sensors.py         # 台架矢量阵列与铯光泵标量观测
│  ├─ features.py        # 源模型/线圈模型特征
│  ├─ models.py          # 基础模型、有限残差修正、航前迁移、线圈响应模型
│  ├─ calibration.py     # 场景隔离的台架/航前标定和各自留出验证
│  ├─ controller.py      # 不使用铯光泵反馈的固化前馈与驱动预加重
│  ├─ environment.py     # 航迹、姿态、地磁和外部磁目标
│  ├─ experiment.py      # 完整实验编排、指标、CSV/JSON 导出
│  ├─ robustness.py      # 多种子/多漂移稳健性矩阵
│  ├─ validation.py      # 旧/新源模型和源漂移验证矩阵
│  ├─ preset_validation.py # 四档界面工况验证
│  ├─ field_validation.py # <1 nT高机动多种子验证与消融
│  └─ webapp.py          # 本地界面 API
├─ static/               # 无外部 CDN 的 HTML/CSS/Canvas 界面
├─ configs/default.json  # 默认可复现实验配置
├─ docs/EXPERIMENT_PROTOCOL.md
├─ tests/
├─ run.py
├─ simulate_cli.py
├─ robustness_cli.py
├─ validate_upgrade.py
├─ validate_field_goal.py
└─ validate_presets.py
```

各模块以明确数组接口连接。后续可单独把偶极源替换成有限元查表、把岭回归替换成其他预估模型，或把理想线圈矩阵替换成真实台架标定数据，而不用重写界面和实验编排。

## 默认物理假设

- 4 类舱内干扰源用带动态滞后、温度项、二次项和交叉耦合的磁偶极子描述；
- 铯光泵敏感体积是体坐标原点，内部坐标约定为 `+X` 向艏、`+Y` 向右舷、`+Z` 向上；所有等效源坐标均以米为单位；
- 台架高精度装置为中心参考点加 `±X/±Y/±Z` 的7点对称三轴阵列；
- 高机动研究模式采用 3 个近似均匀场通道和完整 5 自由度梯度通道；界面保留理论上限、温和工程误差、较强扰动和高机动越限验证四档，默认使用高机动档；
- 梯度保护判据使用对称无散梯度张量的最大绝对主值，而不是只检查 `Gxx`；656 nT/m 作为保守信号退化预警，10000 nT/m 作为额定容限；
- 航行阶段拆除高精度阵列，仅根据设备状态和温度执行前馈；
- 航前迁移训练、航前迁移留出、正式台架留出和航行评估互不混用；航前对基线锚点与当前锚点作差，只识别传递变化，不重训基础模型；
- 铯光泵只观测总磁场模长，不参与参数更新或反馈控制；
- 磁目标只进入最终物理叠加，不进入预估模型或控制器。

## 验证与边界

运行自动检查：

```powershell
pytest -q
```

详细数据隔离、对照设计、筛选指标和实物替换路线见 [严格仿真实验流程](docs/EXPERIMENT_PROTOCOL.md)，本轮结果见 [高机动梯度越限验证](docs/HIGH_MANEUVER_VALIDATION.md)，模型升级见 [源预估模型升级验证报告](docs/UPGRADE_VALIDATION_REPORT.md)，升级前结果留作 [原三档工况历史基线](docs/PRESET_VALIDATION_REPORT.md)。仿真结果只能证明给定模型与参数范围内的可行性，不能直接证明海试性能；尤其不能把“15倍等效磁矩压力”解释成某一实艇的已测磁场。
