# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations

Usage example: python ./batch_load_scannet_data.py
"""
import os
import sys
import datetime
import numpy as np
from vdetr.scannet.load_scannet_data import export
import pdb
import argparse

SCANNET_DIR = '/ps/project/datasets/ScanNet/scannet/scans/'
TRAIN_SCAN_NAMES = [line.rstrip() for line in open('votenet/scannet/meta_data/scannet_train.txt')]
LABEL_MAP_FILE = 'votenet/scannet/meta_data/scannetv2-labels.combined.tsv'
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
ALL_SCANNET_IDS = np.array(range(1, 41))
SMALL_SCANNET_IDS = [i for i in ALL_SCANNET_IDS if i not in OBJ_CLASS_IDS and i not in [1, 2]] # remove wall and floor

SELECT_CLASS_IDS ={
    "all": ALL_SCANNET_IDS,
    "large": OBJ_CLASS_IDS,
    "small": SMALL_SCANNET_IDS,
}

def export_one_scan(scan_name, output_filename_prefix, select_object_type):
    mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.   
    mesh_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = \
        export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None)

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask,:]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]

    num_instances = len(np.unique(instance_labels))
    print('Num of instances: ', num_instances)


    bbox_mask = np.in1d(instance_bboxes[:,-1], SELECT_CLASS_IDS[select_object_type])
    instance_bboxes = instance_bboxes[bbox_mask,:]
    print('Num of care instances: ', instance_bboxes.shape[0])

    N = mesh_vertices.shape[0]

    np.save(output_filename_prefix+'_vert.npy', mesh_vertices)
    np.save(output_filename_prefix+'_sem_label.npy', semantic_labels)
    np.save(output_filename_prefix+'_ins_label.npy', instance_labels)
    np.save(output_filename_prefix+'_bbox.npy', instance_bboxes)

def batch_export(select_object_type, output_folder):
    if not os.path.exists(output_folder):
        print('Creating new data folder: {}'.format(output_folder))
        os.mkdir(output_folder)
        
    for scan_name in TRAIN_SCAN_NAMES:
        print('-'*20+'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = os.path.join(output_folder, scan_name)
        if os.path.isfile(output_filename_prefix+'_vert.npy'):
            print('File already exists. skipping.')
            print('-'*20+'done')
            continue
        try:            
            export_one_scan(scan_name, output_filename_prefix, select_object_type)
        except:
            print('Failed export scan: %s'%(scan_name))            
        print('-'*20+'done')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--select_object_type", type=str, default="large", choices=["large", "small", "all"])
    parser.add_argument("--output_folder", type=str, default="votenet/scannet/scannet_train_detection_data")
    args = parser.parse_args()

    output_folder = args.output_folder + f'_{args.select_object_type}_objects'
    batch_export(select_object_type = args.select_object_type,
                 output_folder = output_folder)
