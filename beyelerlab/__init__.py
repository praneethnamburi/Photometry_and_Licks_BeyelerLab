from collections import Counter
import numpy as np

class Bout:
    def __init__(self, timestamps, th=10):
        self._timestamps = np.asarray(timestamps)
        self.th = th

    def __call__(self):
        return list(self._timestamps)
    
    def __len__(self):
        return len(self._timestamps)

    def bout_classes(self):
        timestamps = self._timestamps
        bout_list = np.zeros(timestamps.size, dtype=int)
        bout_list[0] = 1
        curr_bout = 1
        for sample_count, (prev_stamp, this_stamp) in enumerate(zip(timestamps[:-1], timestamps[1:])):
            if this_stamp - prev_stamp > self.th:
                curr_bout += 1
            bout_list[sample_count+1] = curr_bout
        return bout_list
    
    def bout_start_idx(self):
        return [0] + list(np.where(np.diff(self.bout_classes()) == 1)[0] + 1)
    
    def bout_end_idx(self):
        return list(np.where(np.diff(self.bout_classes()) == 1)[0]) + [len(self)-1]

    def bout_start(self):
        return self._timestamps[self.bout_start_idx()]
    
    def bout_end(self):
        return self._timestamps[self.bout_end_idx()]
    
    def bout_dur(self):
        return self.bout_end() - self.bout_start()
    
    def bout_gap_dur(self):
        return self.bout_start()[1:] - self.bout_end()[:-1]
    
    def n_bouts(self):
        return len(set(self.bout_classes()))
    
    def n_events_per_bout(self):
        return list(Counter(self.bout_classes()).values())
