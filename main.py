import gzip, lzma, zlib, ffmpeg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from functools import partial
from collections import Counter
import glob, os
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class CompressionKNN:
    def __init__(self, image_folder, n_neighbors=3):
        self.images = [f for f in glob.glob(os.path.join(image_folder, '*')) if os.path.isfile(f)]
        self.labels = [os.path.basename(f).split('.')[0] for f in self.images]
        self.indices = np.arange(len(self.labels))
        self.n_neighbors = n_neighbors
        
        self.clfs = {}
        self._create_clfs()
    
    # ====== Compression functions ======
    @staticmethod
    def compress_jpeg(im):
        buf = BytesIO()
        im.save(buf, "JPEG")
        buf.seek(0)
        return buf.read()
    
    @staticmethod
    def compress_jpeg2000(im):
        buf = BytesIO()
        im.save(buf, "JPEG2000")
        buf.seek(0)
        return buf.read()
    
    @staticmethod
    def compress_png(im):
        buf = BytesIO()
        im.save(buf, "PNG")
        buf.seek(0)
        return buf.read()
    
    # ====== Utilities ======
    @staticmethod
    def concat_imgs(img1, img2):
        new_img = Image.new('L', (img1.width, img1.height + img2.height))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (0, img1.height))
        return new_img
    
    # ====== Distance functions ======
    def ncd_dist(self, img1_path, img2_path, compressor=lzma):
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        concat_img = self.concat_imgs(img1, img2)
        
        if compressor in [lzma, gzip, zlib]:
            img1 = img1.tobytes()
            img2 = img2.tobytes()
            concat_img = concat_img.tobytes()
            
        img1_comp = compressor.compress(img1) if hasattr(compressor, "compress") else compressor(img1)
        img2_comp = compressor.compress(img2) if hasattr(compressor, "compress") else compressor(img2)
        concat_comp = compressor.compress(concat_img) if hasattr(compressor, "compress") else compressor(concat_img)
        
        return (len(concat_comp) - min(len(img1_comp), len(img2_comp))) / max(len(img1_comp), len(img2_comp))
    
    @staticmethod
    def video_compression(img1_path, img2_path, compressor='mpeg1video'):
        imgs = ffmpeg.concat(
            ffmpeg.input(img1_path, format='image2', vcodec='gif'),
            ffmpeg.input(img2_path, format='image2', vcodec='gif'),
        )
        stream = imgs.output('pipe:', format='rawvideo', vcodec=compressor)
        stream = stream.global_args('-loglevel', 'error').run_async(pipe_stdout=True)
        return len(stream.stdout.read())
    
    def video_dist(self, img1_path, img2_path, compressor='mpeg1video'):
        Cx_y = self.video_compression(img1_path, img2_path, compressor)
        Cy_x = self.video_compression(img2_path, img1_path, compressor)
        Cx_x = self.video_compression(img1_path, img1_path, compressor)
        Cy_y = self.video_compression(img2_path, img2_path, compressor)
        return (Cx_y + Cy_x)/(Cx_x + Cy_y) - 1
    
    # ====== Sklearn-compatible distance ======
    def custom_dist_sklearn(self, a, b, compressor='mpeg1'):
        comp_map = {
            'gzip': partial(self.ncd_dist, compressor=gzip),
            'lzma': partial(self.ncd_dist, compressor=lzma),
            'mpeg1': partial(self.video_dist, compressor='mpeg1video'),
            'jpeg': partial(self.ncd_dist, compressor=self.compress_jpeg),
            'jpeg2': partial(self.ncd_dist, compressor=self.compress_jpeg2000),
            'png': partial(self.ncd_dist, compressor=self.compress_png),
            'h264': partial(self.video_dist, compressor='h264')
        }
        img1_path = self.images[a.item()]
        img2_path = self.images[b.item()]
        return comp_map[compressor](img1_path, img2_path)
    
    # ====== Create classifiers ======
    def _create_clfs(self):
        compressors = ['mpeg1','h264','gzip','lzma','jpeg','jpeg2','png']
        for comp in compressors:
            n = 1 if comp=='h264' else self.n_neighbors
            self.clfs[comp] = KNeighborsClassifier(
                n_neighbors=n,
                algorithm='brute',
                metric=self.custom_dist_sklearn,
                metric_params={'compressor': comp}
            )
    
    # ====== Cross-validation ======
    def cv_score(self, compressor='gzip', cv=None):
        clf = self.clfs[compressor]
        if cv is None:
            cv = LeaveOneOut()
        return cross_val_score(clf, self.indices.reshape(-1,1), self.labels, scoring='accuracy', cv=cv)
    
    # ====== Predict and plot ======
    def predict_and_plot(self, compressor='jpeg2', test_idx=0):
        clf = self.clfs[compressor]
        Xtrain = np.delete(self.indices, test_idx)
        ytrain = np.delete(self.labels, test_idx)
        clf.n_neighbors = min(clf.n_neighbors, len(Xtrain))
        clf.fit(Xtrain.reshape(-1,1), ytrain)
        
        d, l = clf.kneighbors([[self.indices[test_idx]]], n_neighbors=clf.n_neighbors, return_distance=True)
        nn = Xtrain[l].ravel()
        d = d.ravel()
        
        fig, axs = plt.subplots(1, clf.n_neighbors+1, figsize=(15, 5))
        axs = axs.ravel()
        
        img = Image.open(self.images[test_idx]).convert('RGB')
        axs[0].imshow(img)
        axs[0].set_title('Query')
        axs[0].axis('off')
        
        for i, k in enumerate(nn):
            img = Image.open(self.images[k]).convert('RGB')
            axs[i+1].imshow(img)
            axs[i+1].set_title(f'Neighbor {i+1}\nd={d[i]:.3f}')
            axs[i+1].axis('off')
        
        preds = np.array(self.labels)[nn[:clf.n_neighbors]]
        pred = Counter(preds.flat).most_common(1)[0][0]
        print(f'True label: {self.labels[test_idx]}')
        print(f'Predicted label: {pred}')
