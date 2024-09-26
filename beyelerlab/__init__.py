from collections import Counter
from copy import copy

import numpy as np
import pandas as pd

import pntools as pn
from pntools import sampled

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
        if timestamps.size == 0:
            return []
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
    
    def average_dur_per_bout(self):
        return self.bout_dur()/(np.asarray(self.n_events_per_bout())-1)


class Dataset(pn.FileManager):
    def __init__(self, base_dir, **kwargs):
        # default parameters
        p = copy(kwargs)

        p['file_types'] = kwargs.pop('file_types', ('lick_times', 'isos', 'GCaMP'))
        p['use_names_from'] = kwargs.pop('use_names_from', 'GCaMP')
        p['col_selector'] = kwargs.pop('col_selector', 'Mouse') # column selector for mouse names
        p['target_sr'] = kwargs.pop('target_sr', 30)
        p['mouse_col_offset'] = kwargs.pop('mouse_col_offset', 11)
        p['bout_th'] = kwargs.pop('bout_th', 10)

        super().__init__(base_dir)
        for file_type in p['file_types']:
            self.add(file_type, f'*{file_type}*.csv')
            assert len(self[file_type]) == 1
        
        self.mouse_names = get_mouse_names(self[p['use_names_from']][0], p['col_selector'])
        self.lick_times = read_lick_file(self['lick_times'][0], self.mouse_names, p['mouse_col_offset'], p['bout_th'])
        self.isos = read_sig_file(self['isos'][0], p['target_sr'])
        t_min = self.isos[self.mouse_names[0]].t_start()
        t_max = self.isos[self.mouse_names[0]].t_end()
        for file_type in p['file_types'][2:]:
            setattr(self, file_type, read_sig_file(self[file_type][0], p['target_sr'], t_min, t_max))
        
        # sometimes there is an extra sample in the isos signal, and this code forcefully equalizes the number of samples across modalities (e.g. isos, GCaMP)
        n_samples = {k: [] for k in self.mouse_names}
        for file_type in p['file_types'][1:]:
            for mouse_name in self.mouse_names:
                n_samples[mouse_name].append(len(getattr(self, file_type)[mouse_name]))
        
        for mouse_name in self.mouse_names:
            n = min(n_samples[mouse_name])
            for file_type in p['file_types'][1:]:
                x = getattr(self, file_type)[mouse_name]
                x._sig = x._sig[:n]
                
        self.params = pn.dotdict(p)
        self.t_lim = (t_min, t_max)
    
    def get(self, key):
        assert key in self.mouse_names
        ret = pn.dotdict()
        for file_type in self.params['file_types']:
            ret[file_type] = getattr(self, file_type)[key]
        return ret
    
    __call__ = get


def get_mouse_names(fname, col_selector='Mouse'):
    if isinstance(fname, str):
        df = pd.read_csv(fname)
    else:
        assert isinstance(fname, pd.DataFrame)
        df = fname
    m_names = [x for x in df.columns.values.tolist() if col_selector in x]
    return m_names

def read_sig_file(sig_file, target_sr=30, t_min=None, t_max=None):
    df = pd.read_csv(sig_file)
    m_names = get_mouse_names(df)
    t = df['Timestamp'].to_numpy()
    m_data = {}
    for m_name in m_names:
        m_data[m_name] = sampled.uniform_resample(t, df[m_name].to_numpy(), target_sr, t_min, t_max)
    return m_data

def read_lick_file(lick_file, mouse_names=(1, 2, 3, 4), mouse_col_offset=11, bout_th=10):
    df = pd.read_csv(lick_file).to_numpy()
    t = df[:, 0]
    m_licktimes = dict()
    for mouse_count, mouse_name in enumerate(mouse_names):
        m_licktimes[mouse_name] = Bout(t[df[:, mouse_col_offset+mouse_count].astype(bool)], bout_th)
    return m_licktimes

if __name__ == '__main__':
    Dataset(r'C:\Users\prane\Downloads\PhotoM groupe 1')
