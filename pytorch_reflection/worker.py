# -*- coding: utf-8 -*-

import numpy as np

from dataset.workers import Worker, WorkerType
from dataset.images import Label


class LabelReorderer(Worker):

    message = 'reorder_label'
    worker_type = WorkerType.ADDON

    def process(self, *images):
        results = list()
        for image in images:
            if isinstance(image, Label):
                label_vals = set(image.label_info.labels.values())
                pair_vals = set(np.unique(image.label_info.pairs))
                single_labels = list(label_vals - pair_vals)
                masks = list()
                for p1, p2 in image.label_info.pairs:
                    masks.append(image.data == p1)
                    masks.append(image.data == p2)
                for s in single_labels:
                    mask = image.data == s
                    masks.append(mask)
                data = np.vstack(masks).astype(np.float32)
                results.append(image.update(data, self.message))
            else:
                results.append(image)
        return results
