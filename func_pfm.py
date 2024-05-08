'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
import numpy as np
import sys
import re


############################################################################
#  This file is part of the 4D Light Field Benchmark.                      #
#                                                                          #
#  This work is licensed under the Creative Commons                        #
#  Attribution-NonCommercial-ShareAlike 4.0 International License.         #
#  To view a copy of this license,                                         #
#  visit http://creativecommons.org/licenses/by-nc-sa/4.0/.                #
#                                                                          #
#  Authors: Katrin Honauer & Ole Johannsen                                 #
#  Contact: contact@lightfield-analysis.net                                #
#  Website: www.lightfield-analysis.net                                    #
#                                                                          #
#  The 4D Light Field Benchmark was jointly created by the University of   #
#  Konstanz and the HCI at Heidelberg University. If you use any part of   #
#  the benchmark, please cite our paper "A dataset and evaluation          #
#  methodology for depth estimation on 4D light fields". Thanks!           #
#                                                                          #
#  @inproceedings{honauer2016benchmark,                                    #
#    title={A dataset and evaluation methodology for depth estimation on   #
#           4D light fields},                                              #
#    author={Honauer, Katrin and Johannsen, Ole and Kondermann, Daniel     #
#            and Goldluecke, Bastian},                                     #
#    booktitle={Asian Conference on Computer Vision},                      #
#    year={2016},                                                          #
#    organization={Springer}                                               #
#    }                                                                     #
#                                                                          #
############################################################################
    
def write_pfm(data, fpath, scale=1, file_identifier=b'Pf', dtype="float32"):
    ## PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html
    # data: 要写入文件的图像数据，为一个二维（灰度图）或三维（彩色图）的NumPy数组
    # fpath: 输出PFM文件的路径
    # scale: 缩放因子，用于调整数据的范围，其正负性可指示PFM的字节序（正值为大端，负值为小端）
    # file_identifier: 文件标识符，用于区分PFM文件的类型。b'Pf'表示单通道（灰度图），b'PF'表示三通道（彩色图）。

    data = np.flipud(data)
    # 图像翻转
    # PFM格式要求图像数据从左下角开始，而NumPy数组通常是从左上角开始，因此需要上下翻转图像数据
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    # 将图像数据扁平化为一维数组，以便按顺序写入文件
    endianess = data.dtype.byteorder
    print(endianess)

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        # endianess == "<"  说明为小端字节序
        # endianess == "="  说明与系统的本地字节序相同
        # sys.byteorder     返回系统的本地字节序，"little"为小端
        scale *= -1
        # 若原图为小端字节序，则将pfm的字节序也改成小端（负值）

    with open(fpath, 'wb') as file:
        file.write((file_identifier))
        file.write(('\n%d %d\n' % (width, height)).encode())
        file.write(('%d\n' % scale).encode())

        file.write(values)
        

        
def read_pfm(fpath, expected_identifier="Pf"):
    ## PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html
    # 默认读取灰度图
    def _get_next_line(f):
        # 往后读取一行，并跳过注释（#开头）
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line
    
    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
            # 解析图像尺寸
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            # 断言：缩放因子不为0，否则抛出异常
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
            # 根据缩放因子正负性得出字节序
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            # 按照字节序读取并重塑为二维数组
            data = np.flipud(data)
            # 翻转数组，以符合numpy读取顺序
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
                # 根据缩放因子调整数据的范围
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data