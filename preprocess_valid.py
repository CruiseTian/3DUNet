import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import config

class LITS_preprocess:
    def __init__(self, raw_dataset_path,fixed_dataset_path, args):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        self.classes = args.n_labels # 分割类别数（只分割肝脏为2，或者分割肝脏和肿瘤为3）
        self.upper = args.upper
        self.lower = args.lower
        self.expand_slice = args.expand_slice  # 轴向外侧扩张的slice数量
        self.size = args.min_slices  # 取样的slice数量
        self.xy_down_scale = args.xy_down_scale
        self.slice_down_scale = args.slice_down_scale

    def fix_data(self):
        if not os.path.exists(self.fixed_path):    # 创建保存目录
            #os.makedirs(join(self.fixed_path,'train-images'))
            #os.makedirs(join(self.fixed_path,'train-labels'))
            os.makedirs(join(self.fixed_path,'val-images'))
            os.makedirs(join(self.fixed_path,'val-labels'))
        # train
        '''file_list = os.listdir(join(self.raw_root_path,'ribfrac-train-images'))
        Numbers = len(file_list)
        print('Total numbers of train samples is :',Numbers)
        for ct_file,i in zip(file_list,range(Numbers)):
            print("==== {} | {}/{} ====".format(ct_file, i+1,Numbers))
            ct_path = os.path.join(self.raw_root_path, 'ribfrac-train-images', ct_file)
            seg_path = os.path.join(self.raw_root_path, 'ribfrac-train-labels', ct_file.replace('image', 'label'))
            new_ct, new_seg = self.process(ct_path, seg_path, classes = self.classes)
            if new_ct != None and new_seg != None:
                sitk.WriteImage(new_ct, os.path.join(self.fixed_path, 'train-images', ct_file))  
                sitk.WriteImage(new_seg, os.path.join(self.fixed_path, 'train-labels', ct_file.replace('image', 'label')))'''
        # val
        file_list = os.listdir(join(self.raw_root_path,'ribfrac-val-images'))
        Numbers = len(file_list)
        print('Total numbers of val samples is :',Numbers)
        for ct_file,i in zip(file_list,range(Numbers)):
            print("==== {} | {}/{} ====".format(ct_file, i+1,Numbers))
            ct_path = os.path.join(self.raw_root_path, 'ribfrac-val-images', ct_file)
            seg_path = os.path.join(self.raw_root_path, 'ribfrac-val-labels', ct_file.replace('image', 'label'))
            new_ct, new_seg = self.process(ct_path, seg_path, classes = self.classes)
            if new_ct != None and new_seg != None:
                sitk.WriteImage(new_ct, os.path.join(self.fixed_path, 'val-images', ct_file))  
                sitk.WriteImage(new_seg, os.path.join(self.fixed_path, 'val-labels', ct_file.replace('image', 'label')))

    def process(self, ct_path, seg_path, classes=None):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        print("Ori shape:",ct_array.shape, seg_array.shape)
        if classes==2:
            # 将金标准中肝脏和肝肿瘤的标签融合为一个
            seg_array[seg_array > 0] = 1
        # 将灰度值在阈值之外的截断掉
        ct_array[ct_array > self.upper] = self.upper
        ct_array[ct_array < self.lower] = self.lower

        # 降采样，（对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale）
        ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale), order=3)
        seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale), order=0)
        # 找到肝脏区域开始和结束的slice，并各向外扩张
        try:
            z = np.any(seg_array, axis=(1, 2))
            start_slice, end_slice = np.where(z)[0][[0, -1]]

            # 两个方向上各扩张个slice
            if start_slice - self.expand_slice < 0:
                start_slice = 0
            else:
                start_slice -= self.expand_slice

            if end_slice + self.expand_slice >= seg_array.shape[0]:
                end_slice = seg_array.shape[0] - 1
            else:
                end_slice += self.expand_slice

            print("Cut out range:",str(start_slice) + '--' + str(end_slice))
            # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
            # if end_slice - start_slice + 1 < self.size:
            #     print('Too little slice，give up the sample:', ct_file)
            #     return None,None
            # 截取保留区域
            ct_array = ct_array[start_slice:end_slice + 1, :, :]
            seg_array = seg_array[start_slice:end_slice + 1, :, :]
        except:
            ct_array = ct_array
            seg_array = seg_array
            print(ct_path)
        print("Preprocessed shape:",ct_array.shape,seg_array.shape)
        # 保存为对应的格式
        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale), ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        
        new_seg = sitk.GetImageFromArray(seg_array)
        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale), ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        return new_ct, new_seg

    def write_train_val_name_list(self):
        '''train_name_list = os.listdir(join(self.fixed_path, "train-images"))
        train_num = len(train_name_list)
        print('the fixed dataset total numbers of train samples is :', train_num)
        random.shuffle(train_name_list)'''

        val_name_list = os.listdir(join(self.fixed_path, "val-images"))
        val_num = len(val_name_list)
        print('the fixed dataset total numbers of val samples is :', val_num)

        '''f = open(join(self.fixed_path, "train_path_list.txt"), 'w')
        for name in train_name_list:
            ct_path = os.path.join(self.fixed_path, 'train-images', name)
            seg_path = os.path.join(self.fixed_path, 'train-labels', name.replace('image', 'label'))
            f.write(ct_path + ' ' + seg_path + "\n")
        f.close()'''

        f = open(join(self.fixed_path, "val_path_list.txt"), 'w')
        for name in val_name_list:
            ct_path = os.path.join(self.fixed_path, 'val-images', name)
            seg_path = os.path.join(self.fixed_path, 'val-labels', name.replace('image', 'label'))
            f.write(ct_path + ' ' + seg_path + "\n")
        f.close()


if __name__ == '__main__':
    raw_dataset_path = './datasets/'
    fixed_dataset_path = './datasets/process/'

    args = config.args 
    tool = LITS_preprocess(raw_dataset_path,fixed_dataset_path, args)
    tool.fix_data()                            # 对原始图像进行修剪并保存
    tool.write_train_val_name_list()      # 创建索引txt文件