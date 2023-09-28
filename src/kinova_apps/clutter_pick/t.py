from shapely.geometry import Polygon, LineString, MultiPolygon, Point
import matplotlib.pyplot as plt
from shapely import plotting, affinity

import cv2
from cv_bridge import CvBridge

import numpy as np

import rospy

from copy import deepcopy

from typing import List, Set, Union

class PropertyPolygon(Polygon):
    _id_to_attrs = {}

    __slots__ = Polygon.__slots__

    color: str
    mask: np.ndarray
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


class ClutterDetector:
    def __init__(self):
        self.bridge = CvBridge()

    def analyze_cluttered_polys(self, clutter_polys) -> Set[Union[List[PropertyPolygon], List[List[PropertyPolygon]]]]:

        can_be_picked_polys: List[PropertyPolygon] = []
        new_clutter_polys: List[List[PropertyPolygon]] = []

        for multi_poly in clutter_polys:
            local_clutter_polys = []
            for rect in multi_poly:

                print(f'poly color: {rect.color}')

                # get the lines of opposite sides of the polygon
                opp_lines_1: List[LineString] = []
                opp_lines_2: List[LineString] = []

                l1 = LineString(
                    [rect.exterior.coords[0], rect.exterior.coords[1]]
                )
                l2 = LineString(
                    [rect.exterior.coords[2], rect.exterior.coords[3]]
                )
                opp_lines_1.append(l1)
                opp_lines_1.append(l2)

                l1 = LineString(
                    [rect.exterior.coords[1], rect.exterior.coords[2]]
                )
                l2 = LineString(
                    [rect.exterior.coords[3], rect.exterior.coords[0]]
                )
                opp_lines_2.append(l1)
                opp_lines_2.append(l2)

                # draw the rect
                cv2.drawContours(
                    image,
                    [np.array(rect.exterior.coords, dtype=np.int32)],
                    0,
                    (0, 0, 0),
                    2,
                )

                ol1_free = True
                # check if each line in opp_lines_1 has min dist from other polygons in multi_poly
                for line in opp_lines_1:
                    # check if perpendicular distance is less than min_clearance
                    line_center = line.centroid

                    # find a point far away from the line center in the direction perpendicular to the line
                    l0 = line.parallel_offset(10, "right")

                    line_center = line.centroid
                    point1 = (
                        l0.centroid
                        - line_center
                    )

                    point2 = (
                        line.parallel_offset(30, "right").centroid
                        - line_center
                    )
                    point3 = (
                        line.parallel_offset(50, "right").centroid
                        - line_center
                    )

                    point4 = (
                        line.parallel_offset(5, "right").centroid
                        - line_center
                    )

                    point5 = (
                        line.parallel_offset(15, "right").centroid
                        - line_center
                    )

                    point6 = (
                        line.parallel_offset(25, "right").centroid
                        - line_center
                    )

                    

                    for other_poly in multi_poly:
                        if other_poly == rect:
                            continue

                        # check if the point is inside the polygon
                        if (
                            other_poly.contains(point1) or other_poly.intersects(point1)
                            or other_poly.contains(point2) or other_poly.intersects(point2)
                            or other_poly.contains(point3) or other_poly.intersects(point3)
                            or other_poly.contains(point4) or other_poly.intersects(point4)
                            or other_poly.contains(point5) or other_poly.intersects(point5)
                            or other_poly.contains(point6) or other_poly.intersects(point6)
                        ):
                            ol1_free = False
                            break

                    if not ol1_free:
                        break

                if ol1_free:
                    print('inside')
                    # draw lines
                    for line in opp_lines_1:
                        cv2.line(
                            image,
                            (
                                int(line.coords[0][0]),
                                int(line.coords[0][1]),
                            ),
                            (
                                int(line.coords[1][0]),
                                int(line.coords[1][1]),
                            ),
                            (0, 0, 255),
                            2,
                        )

                        

                    # find angle parallel to the parallel lines
                    angle = np.arctan2(
                        line.coords[1][1] - line.coords[0][1],
                        line.coords[1][0] - line.coords[0][0],
                    )

                    rect.angle = angle

                    can_be_picked_polys.append(rect)
                    continue

                ol2_free = True
                # check if each line in opp_lines_2 has min dist from other polygons in multi_poly
                for line in opp_lines_2:

                    l0 = line.parallel_offset(10, "right")

                    line_center = line.centroid
                    point1 = (
                        l0.centroid
                        - line_center
                    )
                    point2 = (
                        line.parallel_offset(30, "right").centroid
                        - line_center
                    )
                    point3 = (
                        line.parallel_offset(50, "right").centroid
                        - line_center
                    )
                    point4 = (
                        line.parallel_offset(5, "right").centroid
                        - line_center
                    )

                    point5 = (
                        line.parallel_offset(15, "right").centroid
                        - line_center
                    )

                    point6 = (
                        line.parallel_offset(25, "right").centroid
                        - line_center
                    )

                    for other_poly in multi_poly:
                        if other_poly == rect:
                            continue

                        # check if the point is inside the polygon
                        if (
                            other_poly.contains(point1) or other_poly.intersects(point1)
                            or other_poly.contains(point2) or other_poly.intersects(point2)
                            or other_poly.contains(point3) or other_poly.intersects(point3)
                            or other_poly.contains(point4) or other_poly.intersects(point4)
                            or other_poly.contains(point5) or other_poly.intersects(point5)
                            or other_poly.contains(point6) or other_poly.intersects(point6)
                        ):
                            ol2_free = False
                            break

                    if not ol2_free:
                        break

                if ol2_free:
                    # draw lines
                    for line in opp_lines_2:
                        cv2.line(
                            image,
                            (
                                int(line.coords[0][0]),
                                int(line.coords[0][1]),
                            ),
                            (
                                int(line.coords[1][0]),
                                int(line.coords[1][1]),
                            ),
                            (0, 255, 0),
                            2,
                        )

                    

                    # find angle parallel to the parallel lines
                    angle = np.arctan2(
                        opp_lines_2[0].coords[1][1] -  opp_lines_2[0].coords[0][1],
                         opp_lines_2[0].coords[1][0] -  opp_lines_2[0].coords[0][0],
                    )

                    # plot the angle on the image
                    cv2.arrowedLine(
                        image,
                        (
                            int(rect.centroid.coords[0][0]),
                            int(rect.centroid.coords[0][1]),
                        ),
                        (
                            int(rect.centroid.coords[0][0] + np.cos(angle) * 100),
                            int(rect.centroid.coords[0][1] + np.sin(angle) * 100),
                        ),
                        (0, 0, 255),
                        2,
                    )

                    rect.angle = angle

                    can_be_picked_polys.append(rect)
                    continue

                if not ol1_free and not ol2_free:
                    local_clutter_polys.append(rect)

            if len(local_clutter_polys) > 0:
                new_clutter_polys.append(local_clutter_polys)

        return can_be_picked_polys, new_clutter_polys


    def process_image_with_cubes(
        self, cv_image
    ) -> [bool, List[Polygon], List[Polygon]]:
        # convert image to cv2
        # cv_image = self.bridge.imgmsg_to_cv2(
        #     image, desired_encoding="passthrough"
        # )

        # convert to bgr
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        # get the hsv_ranges from the parameter server
        hsv_ranges = rospy.get_param("/clutter/hsv_ranges")

        # create colors list
        colors = [key for key in hsv_ranges.keys()]

        # Create HSV Image
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # create masks for each color
        masks = []
        for color in colors:
            lower = np.array(hsv_ranges[color]["lower"])
            upper = np.array(hsv_ranges[color]["upper"])
            mask = cv2.inRange(hsv_image, lower, upper)
            masks.append(mask)

        # get polygons for each color
        polygons: dict[str, PropertyPolygon] = {}
        detected_masks = {}
        for i, color in enumerate(colors):
            dmask, polys = self.apply_mask_and_get_masks_polygons(
                cv_image, masks[i], color
            )
            detected_masks[color] = dmask
            polygons[color] = polys

        # combine all polygons
        ps = []
        for color in polygons:
            for p, m in zip(polygons[color], detected_masks[color]):
                p.color = color
                p.mask = m
                ps.append(p)
        
        draw_image = cv_image.copy()

        # check if two or more polygons are close to each other
        cluttered_polygons, free_polygons = self.check_polygons(ps)

        if len(cluttered_polygons) > 0:
            
            can_be_picked_polys, new_clutter_polys = self.analyze_cluttered_polys(cluttered_polygons)

            cluttered_polygons = new_clutter_polys

            free_polygons.extend(can_be_picked_polys)
            
            # draw cluttered polygons
            for multi_poly in cluttered_polygons:
                # convex hull
                poly = MultiPolygon(deepcopy(multi_poly)).convex_hull
                # convert to contour
                polyg_coords = np.array(poly.exterior.coords, dtype=np.int32)
                # draw contour
                cv2.drawContours(draw_image, [polyg_coords], -1, (0, 0, 255), 3)
                cv2.imshow("draw_image", draw_image)
                cv2.waitKey(0)

        dms = []
        for i, poly in enumerate(free_polygons):
            
            dmask = poly.mask
            poly_color = poly.color

            dmask = cv2.cvtColor(dmask, cv2.COLOR_GRAY2BGR)

            # get the contours from the mask
            # convert dmask to 8uc1
            dmask = cv2.cvtColor(dmask, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(
                dmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            # remove small contours
            contours = [c for c in contours if cv2.contourArea(c) > 1000]

            # get the contour with the largest area
            contour = max(contours, key=cv2.contourArea)

            # fit a square to the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # draw the box
            cv2.drawContours(draw_image, [box], 0, (0, 0, 0), 2)

            # annotate the color of the box
            cv2.putText(
                draw_image,
                poly_color,
                (int(rect[0][0]), int(rect[0][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

            # get the center of the box
            center = rect[0]

            # get two points corresponding to same side of the box
            p1 = box[1]
            p2 = box[2]

            side_middle = (p1 + p2) / 2

            if poly.angle is None:

                # get the angle of the box
                angle = np.arctan2(
                    side_middle[1] - center[1], side_middle[0] - center[0]
                )
            else:
                angle = poly.angle

            # draw the pose on the image from box center
            draw_image = cv2.arrowedLine(
                draw_image,
                (int(center[0]), int(center[1])),
                (
                    int(center[0] + np.cos(angle) * 100),
                    int(center[1] + np.sin(angle) * 100),
                ),
                (0, 0, 255),
                2,
            )

            poly.angle = angle

            dms.append(poly)

        # show the image
        cv2.imshow("draw_image", draw_image)
        cv2.waitKey(0)

        return len(cluttered_polygons) > 0, dms, cluttered_polygons

    def check_polygons(
        self, polygons: List[PropertyPolygon]
    ):
        cluttered_polygons: List[List[PropertyPolygon]] = []
        free_polygons: List[PropertyPolygon] = []

        # min_clearance from parameter server
        min_clearance = rospy.get_param("clutter/min_clearance")

        cluttered_poly_indices = []

        # check if two or more polygons are close to each other
        for i, polygon in enumerate(polygons):
            multipolygons = []
            for j, other_polygon in enumerate(polygons):
                if i == j:
                    continue

                if polygon.distance(other_polygon) < min_clearance:
                    if j not in cluttered_poly_indices:
                        multipolygons.append(other_polygon)
                        cluttered_poly_indices.append(j)

            if len(multipolygons) > 0:
                # add polygon to multipolygon
                multipolygons.append(polygon)
                if i not in cluttered_poly_indices:
                    cluttered_poly_indices.append(i)
                # convert to multipolygon
                multipolygon = MultiPolygon(deepcopy(multipolygons))
                # add multipolygon to list if not already present
                for mp in cluttered_polygons:
                    if MultiPolygon(deepcopy(mp)).convex_hull == multipolygon.convex_hull:
                        break
                else:
                    cluttered_polygons.append(multipolygons)


        # get free polygons
        free_poly_indices = [
            i for i in range(len(polygons)) if i not in cluttered_poly_indices
        ]

        free_polygons = [polygons[i] for i in free_poly_indices]

        return cluttered_polygons, free_polygons

    def apply_mask_and_get_masks_polygons(self, cv_image, mask, color):
        """
        apply the color mask on the cv image to detect cubes
        return the masks and shapely polygons for each cube
        """

        # apply mask
        res = cv2.bitwise_and(cv_image, cv_image, mask=mask)

        if color == 'red':
            cv2.imshow('res', res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # find contours for detecting cubes
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # remove small contours
        contours = [c for c in contours if cv2.contourArea(c) > 1000 and cv2.contourArea(c) < 3500]

        if color == 'orange':
            # print contour area
            for c in contours:
                print(cv2.contourArea(c))

        if color == 'red':
            # print contour area
            print('red')
            for c in contours:
                print(cv2.contourArea(c))

        # draw contours
        cv2.drawContours(cv_image, contours, -1, (0, 255, 0), 2)

        cv2.imshow("cv_image", cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # get the pixels lying inside the contours
        masks = []
        polygons = []
        for c in contours:
            # create mask
            mask = np.zeros(cv_image.shape[:2], np.uint8)
            # draw contour
            cv2.drawContours(mask, [c], -1, 255, -1)
            # add mask to list
            masks.append(mask)

            dmask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # get the contours from the mask
            # convert dmask to 8uc1
            dmask = cv2.cvtColor(dmask, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(
                dmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            # remove small contours
            contours = [c for c in contours if cv2.contourArea(c) > 1000]

            # get the contour with the largest area
            contour = max(contours, key=cv2.contourArea)

            # fit a square to the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # convert to shapely polygon
            poly = PropertyPolygon(box)
            polygons.append(poly)

        return masks, polygons


if __name__ == "__main__":
    cd = ClutterDetector()
    image = cv2.imread("/home/batsy/kinova_ros/cluttered_cubes/frame0000.jpg")
    cd.process_image_with_cubes(image)
