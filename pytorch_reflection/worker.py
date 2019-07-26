# -*- coding: utf-8 -*-

import numpy as np

from dataset.workers import Worker
from dataset.images import Label


class LabelReorderer(Worker):

    message = 'reorder_label'

    def process(self, *images):
        results = list()
        for image in images:
            if isinstance(image, Label):
                label_vals = set(image.labels.values())
                pair_vals = set(np.unique(image.pairs))
                single_labels = list(label_vals - pair_vals)
                masks = list()
                for p1, p2 in image.pairs:
                    masks.append(image.data == p1)
                    masks.append(image.data == p2)
                for s in single_labels:
                    mask = image.data == s
                    masks.append(mask)
                    masks.append(mask)
                data = np.vstack(masks).astype(np.float32)
                results.append(image.update(data, self.message))
            else:
                results.append(image)
        return results
