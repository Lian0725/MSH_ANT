# Wang Figure 9 二维相速度反演

该目录保存 Wang et al. (2017) 风格的方位校正二维相速度反演代码，以及 `egf_mid` 的最终绘图结果。

## 代码

- `code/plot_wang_fig9_azimuthal_correction.py`：方位残差 Fourier 校正、校正前后二维反演与 Figure 8/9 绘图入口。
- `code/local_phase_velocity_maps.py`：直线路径矩阵、阻尼/平滑正则化与鲁棒最小二乘二维慢度反演。
- `code/aant_2014_phase_maps.py`：测地距离、路径裁剪等几何辅助函数。
- `code/replot_wang_figure6_colormap.py`：Wang 风格配色、显示平滑和色标辅助函数。
- `code/plot_wang_fig5_fig6_from_disperpicker.py`：由 DisperPicker 测量绘制 Figure 5/6 的二维相速度图。
- `code/run_fig9_resolution_tests.py`、`code/render_fig9_checkerboard_km_comparison.py`：二维反演的棋盘格/分辨率测试与对比渲染。

## 当前结果

- `figures/wang_figure8_style_azimuthal_residual_fit.png`：方位残差及 Fourier 拟合。
- `figures/wang_figure9_style_phase_velocity_maps_azimuthal_corrected.png`：方位校正后的 3、3.5、4 s 相速度图。
- `figures/wang_figure6_vs_figure9_before_after_comparison.png`：校正前后对比图；左侧行标签使用 Times New Roman。

结果对应服务器报告：
`wang_fig9_azimuth_corrected_egf_mid_20260724`。

## 运行依赖与输入

需要 Python、NumPy、SciPy、Matplotlib（直接运行 `local_phase_velocity_maps.py` 时还需要 pandas），以及已准备好的 EGF 相速度测量 CSV。运行入口是 `plot_wang_fig9_azimuthal_correction.py`；其余三个脚本须与入口脚本放在同一目录。对比图行标签的 Times New Roman 字体路径是当前服务器专用路径，迁移到其他主机时请按本机字体位置调整该常量。
