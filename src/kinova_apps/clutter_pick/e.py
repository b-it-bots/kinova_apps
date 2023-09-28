#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2
from numpy import ndarray

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import shapely
from shapely.geometry import Polygon, MultiPolygon

class CPolygon(Polygon):
    def __init__(self, exterior, holes=None, color=None, mask=None, angle=None):
        super().__init__(exterior, holes)
        self._color = color
        self._mask = mask
        self._angle = angle

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value

def main():
    """Extract a folder of images from a rosbag.
    """

    bag = rosbag.Bag('/home/batsy/kinova_ros/real_obj_vary_height.bag', "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw']):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        cv2.imwrite(os.path.join('/home/batsy/kinova_ros/ext_imgs_vary_ht', "frame%06i.png" % count), cv_img)

        count += 1

    bag.close()

    return

def test():
    from shapely.geometry import Polygon

    class PropertyPolygon(Polygon):
        _id_to_attrs = {}

        __slots__ = Polygon.__slots__

        color: str
        mask: ndarray
        angle: float

        def __init__(self, exterior, holes=None, color=None, mask=None, angle=None):
            self._id_to_attrs[id(self)] = dict(color=color, mask=mask, angle=angle)

        def __new__(cls, exterior, holes=None, *args, **kwargs):
            poly = super().__new__(cls, exterior, holes)
            poly.__class__ = cls
            return poly
        
        def __del__(self):
            del self._id_to_attrs[id(self)]

        def __getattr__(self, key):
            try:
                return PropertyPolygon._id_to_attrs[id(self)][key]
            except KeyError as e:
                raise AttributeError(key) from e
            
        def __setattr__(self, key, value):
            if key in PropertyPolygon.__slots__:
                super().__setattr__(key, value)
            else:
                PropertyPolygon._id_to_attrs[id(self)][key] = value
            
        def __str__(self):
            return f'{self.color}'
        
    class PropertyMultiPolygon(MultiPolygon):
        
        def __new__(self, polygons=None):
            if not polygons:
                # Allow creation of empty multipolygons, to support unpickling
                # TODO better empty constructor
                return shapely.from_wkt("MULTIPOLYGON EMPTY")
            elif isinstance(polygons, MultiPolygon):
                return polygons

            polygons = getattr(polygons, "geoms", polygons)
            polygons = [
                p
                for p in polygons
                if p and not (isinstance(p, PropertyPolygon) and p.is_empty)
            ]

            L = len(polygons)

            # Bail immediately if we have no input points.
            if L == 0:
                return shapely.from_wkt("MULTIPOLYGON EMPTY")

            # This function does not accept sequences of MultiPolygons: there is
            # no implicit flattening.
            if isinstance(polygons[0], MultiPolygon):
                raise ValueError("Sequences of multi-polygons are not valid arguments")

            subs = []
            for i in range(L):
                ob = polygons[i]
                if not isinstance(ob, PropertyPolygon):
                    shell = ob[0]
                    holes = ob[1]
                    p = PropertyPolygon(shell, holes)
                else:
                    p = ob
                subs.append(p)

            return shapely.multipolygons(subs)
        

   # Create PropertyPolygon instances
    polygon1 = PropertyPolygon([(0, 0), (0, 1), (1, 1), (1, 0)], color='red')
    polygon2 = PropertyPolygon([(1, 0), (1, 1), (2, 1), (2, 0)], color='blue')

    # Create a PropertyMultiPolygon instance and add PropertyPolygon objects
    multi_polygon = PropertyMultiPolygon([polygon1, polygon2])

    # Access the polygons within the multi-polygon
    for polygon in multi_polygon.geoms:
        print(f'Color: {polygon.color}')

if __name__ == '__main__':
    test()