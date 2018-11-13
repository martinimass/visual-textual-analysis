#!/usr/bin/env python
from __future__ import print_function
import sys
import os

path_results="../results/textual/"

class TextBox:
    def __init__(self, name, x, y, w, h):
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return 'TB(name={}, x={}, y={})'.format(self.name, self.x, self.y)

    def __repr__(self):
        return str(self)

    def is_neighbour(self, other, delta):
        if abs(self.y - other.y) > delta:
            return False
        return True

    def neighbours(self, others, delta):
        return [neighbour for neighbour in others if self.is_neighbour(neighbour, delta)]


class TextLiner:
    """
    One per image
    """
    def __init__(self, boxes=[], linedelta_px=5):
        self.boxes = boxes
        self.lines = {}
        self.delta = linedelta_px

    def produce_lines(self):
        # sort boxes by y, reversed for popping
        yboxes  = sorted(self.boxes, key=lambda b: b.y, reverse=True)
        current_line_number = 0
        lines = {}
        while yboxes:
            current_ybox = yboxes.pop()
            # first find if our ybox belongs to a line
            for lindex, line_boxes in lines.items():
                if current_ybox.neighbours(line_boxes, self.delta):     # we belong to this line
                    lines[lindex].append(current_ybox)
                    break
            else:
                # no line wants us as neighbour
                # --> we create a new line for ourselves
                lines[current_line_number] = [
                    current_ybox]
                current_line_number += 1
        # fix the x
        for lindex, line_boxes in lines.items():
            line_boxes.sort(key=lambda b: b.x)
        return lines


class BatchTextLiner:
    def __init__(self, tipo, delta_y):
        self.tipo=tipo
        self.folder = path_results+tipo+"1"
        self.out_folder= path_results+tipo+"2"
        self.delta = delta_y

    def process_folder(self):
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
        cont=1
        for filn in [f for f in os.listdir(self.folder) if f.endswith('.crop_coords.txt')]:
            print('{} - {} - Processing {}'.format(cont,self.tipo,filn))
            cont+=1
            filp = os.path.join(self.folder, filn)
            filp2 = os.path.join(self.out_folder, filn)
            with open(filp, 'rt') as f:
                lines = [line.strip() for line in f.readlines()]
            boxes = []
            for line in lines:
                crop_name, x1, y1, x2, y2 = line.split(',')
                boxes.append(TextBox(crop_name, int(x1), int(y1), int(x2), int(y2)))
            tl = TextLiner(boxes=boxes, linedelta_px=self.delta)
            out_filp = filp2.replace('crop_coords', 'crops_in_order')
            with open(out_filp, 'wt') as f:
                for lindex, boxes in tl.produce_lines().items():
                    for box in boxes:
                        f.write('{},{},{},{},{}\n'.format(box.name, box.x, box.y, box.w, box.h))

if __name__ == '__main__':
	delta = 5
	for folder in ["train","test"]:
		btl = BatchTextLiner(tipo=folder, delta_y=delta)
		btl.process_folder()
		print('Folder {} OK'.format(folder))
