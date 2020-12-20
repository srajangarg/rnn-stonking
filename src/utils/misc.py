import copy
import inspect

# import cv2
import numpy as np
import yaml

from .attr_dict import nested_attr_dict


def load_config(config_file):
    cfg = None
    with open(config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return nested_attr_dict(cfg)

def get_default_kwargs(func):
    """ Returns default kwargs for function `func` as a dict
    """
    sig = inspect.signature(func)
    kwargs = {}
    for pname,defval in dict(sig.parameters).items():
        if defval.default != inspect.Parameter.empty:
            kwargs[pname] = copy.deepcopy(defval.default)
    return kwargs

def write_video(fname, frames, fps=25, size=None, codec='MJPG'):
    if size is None:
        size = (frames[0].shape[1],frames[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(fname, fourcc, fps, size)
    for frame in frames:
        out.write(frame)
    out.release()

def read_video(fname, frame_nums=None, start=0, num_read=None):
    video_cap = cv2.VideoCapture(fname)
    assert(video_cap.isOpened())

    if num_read is None:
        num_read = vid_size-start

    vid_w = video_cap.get(3)
    vid_h = video_cap.get(4)
    vid_fps = video_cap.get(5)
    vid_size = video_cap.get(7)
    assert(vid_size>0)

    frames = []
    if frame_nums is None:
        assert(start<vid_size)
        video_cap.set(1, start)
        read = 0
        while(video_cap.isOpened()):
            if read >= num_read:
                break
            ret, frame = video_cap.read()
            read += 1
            if ret != True:
                break
            assert(frame.shape[:2] == (vid_h, vid_w))
            frames.append(frame)
        assert(len(frames) == num_read)
    else:
        for n in frame_nums:
            assert(n<vid_size)
            video_cap.set(1, n)
            ret, frame = video_cap.read()
            frames.append(frame)

    return frames, vid_fps

def show_video(frames, fps):
    for frame in frames:
        cv2.imshow('Frame',frame)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'): # Press Q to exit
            break
    cv2.destroyWindow('Frame')

def frames_generator(video_cap, step=1, start=0, num_read=None, aggregate=None):

    if aggregate is not None:
        assert(start==0)
        assert(num_read is None)

    assert(video_cap.isOpened())
    video_cap.set(1,int(start))
    if num_read is None:
        num_read = (int(video_cap.get(7))-start)//step

    aggregated = []
    sstep = 0
    read = 0
    while(video_cap.isOpened()):
        if read >= num_read:
            break
        ret,frame = video_cap.read()
        sstep += 1
        if ret is True:
            if sstep == 1:
                if aggregate is None:
                    yield frame
                    read += 1
                elif len(aggregated)==aggregate:
                    yield aggregated
                    aggregated = []
                else:
                    aggregated.append(frame)

            if sstep == step:
                sstep = 0
        else:
            break


def read_matrix_file(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [[float(f) for f in x.strip().split()] for x in content]
    return np.array(content)

def _is_char_digit(c):
    return 49 <= ord(c) <= 57
def _is_char_az(c):
    return 97 <= ord(c) <= 122
def _is_char_AZ(c):
    return 65 <= ord(c) <= 90
def _is_char_alphabet(c):
    return _is_char_az(c) or _is_char_AZ(c)
def _process_string(s):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            try: # Array of float
                return np.array([float(x.strip()) for x in s.split()])
            except:
                return s

def read_info_file(fname):
    # Processes text files where each lines is a variable assignemnt.
    # For example:
    # ```
    # length = 123
    # pi = 3.14
    # name = Shubham
    # list = 0 0.1 0.2 0.3
    # ```
    # Returns a dictionary:
    # {
    #   'length': 123,
    #   'pi': 3.14,
    #   'name': 'Shubham',
    #   'list': np.array([0.0, 0.1, 0.2, 0.3])
    # }
    with open(fname) as f:
        content = f.readlines()
    content = [ x.strip().split('=') for x in content]
    content = [[x.strip() for x in l] for l in content]
    keys = [l[0] for l in content]
    values = [_process_string(l[1]) for l in content]
    return dict(zip(keys, values))


def read_file_as_list(fname):
    with open(fname, 'r') as f:
        ll = f.readlines()
    return [l.strip() for l in ll if len(l.strip())>0]

def sec_to_hms(ss):
    ss = int(ss)
    mm = ss//60
    hh = mm//60
    mm = mm%60
    ss = ss%60
    return hh,mm,ss

class RecentList():
    def __init__(self,max_size=10):
        assert(max_size > 0)
        self.max_size = max_size
        self.list = []
        self.csum = 0
    def append(self, x):
        if len(self.list) == self.max_size:
            self.csum -= self.list[0]
            self.list.pop(0)
        x = float(x)
        self.list.append(x)
        self.csum += x
    def average(self):
        return self.csum/(len(self.list) + 1e-12)

class AverageMeter():
    def __init__(self, max_size=10):
        self.csum = 0
        self.count = 0
        self.recent = RecentList(max_size=max_size)
    def append(self, x):
        self.csum += float(x)
        self.count += 1
        self.recent.append(x)
    def average(self):
        return self.csum/(self.count + 1e-12)
    def recent_average(self):
        return self.recent.average()

def plot_triangles(triangles):
    """
        triangles: nx3x2 n triangles with coordinares in [-1,1]

    """
    lines0 = triangles[:,[0,1],:]
    lines1 = triangles[:,[2,1],:]
    lines2 = triangles[:,[0,2],:]
    lines = np.concatenate((lines0,lines1,lines2), axis=0)
    plot_lines(lines)

def plot_lines(lines):
    """
        lines: nx2x2 n lines with coordinares in [-1,1]

    """
    import matplotlib.pyplot as plt
    for (x0,y0),(x1,y1) in lines:
        plt.plot([x0,x1],[y0,y1])
    plt.show()
