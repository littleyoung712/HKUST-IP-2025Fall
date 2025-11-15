1.因子计算文件：cost_score & trend_calculation
* 测试需要将因子计算文件和score生成文件放在同一级文件夹中，score生成文件需要import因子计算文件

2. score计算文件： tr_sig_gen
因子计算文件中需要在main函数中修改对应的folder_name和index，tr_sig_gen生成分标的时序文件，生成的所有df在同文件夹下的output中
D:\Python\HKUST_IP2\sp500_processedOHLC_CRSP\output_data
D:\Python\HKUST_IP2\tw50_processedOHLC