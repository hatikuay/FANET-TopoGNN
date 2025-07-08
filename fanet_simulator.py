import numpy as np
import networkx as nx
from scipy.spatial import distance, ConvexHull
import gudhi as gd
from persim import PersImage
from persim import PersistenceImager
import h5py
import os
from tqdm import tqdm

# 1. Parametreler
class SimulationConfig:
    def __init__(self):
        self.area = (0, 1000, 0, 1000)  # x_min, x_max, y_min, y_max
        self.N_list = [10, 20, 30, 40, 50]  # UAV sayıları
        self.T = 1000  # Toplam simülasyon süresi (saniye)
        self.dt = 0.1  # Zaman adımı (saniye)
        self.snapshots_per_run = int(self.T / self.dt)
        self.r_base = 300  # Temel iletişim yarıçapı (metre)
        self.rho_thr = 1.5e-4  # Yoğunluk eşiği (UAV/m²)
        self.kappa = 1.6  # Adaptif yarıçap çarpanı
        self.epsilon_max = 600  # Maksimum filtrasyon değeri (metre)
        self.pixels = (20, 20)  # Persistence Image boyutu
        self.sigma = 20  # Gaussian kernel standart sapması
        self.num_runs = 5  # Her konfigürasyon için çalıştırma sayısı
        self.mobility_models = ['RWP', 'GM']  # Hareket modelleri

# 2. Hareket Modelleri
class RandomWaypoint:
    def __init__(self, area, N, speed_min=10, speed_max=30, pause_max=0):
        self.area = area
        self.N = N
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.pause_max = pause_max
        self.positions = self.initialize()
        
    def initialize(self):
        x_min, x_max, y_min, y_max = self.area
        return np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(self.N, 2))
    
    def step(self, dt):
        new_positions = np.copy(self.positions)
        # Basitleştirilmiş hareket: Rastgele yön ve hız
        speeds = np.random.uniform(self.speed_min, self.speed_max, self.N)
        angles = np.random.uniform(0, 2*np.pi, self.N)
        new_positions[:, 0] += speeds * np.cos(angles) * dt
        new_positions[:, 1] += speeds * np.sin(angles) * dt
        
        # Sınır kontrolü
        x_min, x_max, y_min, y_max = self.area
        new_positions[:, 0] = np.clip(new_positions[:, 0], x_min, x_max)
        new_positions[:, 1] = np.clip(new_positions[:, 1], y_min, y_max)
        
        self.positions = new_positions
        return self.positions

class GaussMarkov:
    def __init__(self, area, N, alpha=0.8, mu=20, sigma=5):
        self.area = area
        self.N = N
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.positions = self.initialize()
        self.velocities = np.random.uniform(-1, 1, (N, 2)) * mu
        
    def initialize(self):
        x_min, x_max, y_min, y_max = self.area
        return np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(self.N, 2))
    
    def step(self, dt):
        # Hız güncelleme
        self.velocities = (self.alpha * self.velocities + 
                          (1 - self.alpha) * self.mu +
                          np.sqrt(1 - self.alpha**2) * self.sigma * np.random.normal(size=(self.N, 2)))
        
        # Pozisyon güncelleme
        self.positions += self.velocities * dt
        
        # Sınır kontrolü (yansıtma)
        x_min, x_max, y_min, y_max = self.area
        for i in range(self.N):
            x, y = self.positions[i]
            vx, vy = self.velocities[i]
            
            if x < x_min or x > x_max:
                self.velocities[i, 0] = -vx
                self.positions[i, 0] = np.clip(x, x_min, x_max)
                
            if y < y_min or y > y_max:
                self.velocities[i, 1] = -vy
                self.positions[i, 1] = np.clip(y, y_min, y_max)
                
        return self.positions

import numpy as np
import gudhi as gd
from persim import PersistenceImager

def compute_persistence_image(positions, epsilon_max, pixels, sigma):
    """
    Computes a persistence image from the H0-persistence diagram of the input point cloud.

    Parameters:
    - positions: (N,2) array of point coordinates
    - epsilon_max: float, maximum filtration (edge) length
    - pixels: tuple (px, py), number of pixels along birth and persistence axes
    - sigma: float, Gaussian kernel standard deviation

    Returns:
    - 1D numpy array of length px * py (flattened persistence image)
    """
    # 1. Build Rips complex and simplex tree up to dimension 1
    rc = gd.RipsComplex(points=positions, max_edge_length=epsilon_max)
    st = rc.create_simplex_tree(max_dimension=1)  # 1-skeleton :contentReference[oaicite:0]{index=0}

    # 2. Compute persistence
    persistence = st.persistence()  # list of (dim, (birth, death)) :contentReference[oaicite:1]{index=1}

    # 3. Extract H0 (connected components) and cap infinite deaths
    H0_pairs = []
    for dim, (b, d) in persistence:
        if dim == 0:
            if d == float('inf'):
                d = epsilon_max
            H0_pairs.append((b, d))

    # 4. If there are no H0 features, return zeros
    if not H0_pairs:
        return np.zeros(pixels[0] * pixels[1])

    # 5. Convert to array and set up PersistenceImager
    dgm = np.array(H0_pairs)
    # Determine pixel size so that image spans [0, epsilon_max] in both axes
    pixel_size = epsilon_max / pixels[0]

    pimgr = PersistenceImager(
        birth_range=(0.0, epsilon_max),
        pers_range=(0.0, epsilon_max),
        pixel_size=pixel_size,
        kernel='gaussian',
        kernel_params={'sigma': sigma}
    )  # :contentReference[oaicite:2]{index=2}

    # 6. Transform the diagram into an image (skew=False since we already use (birth, persistence))
    img = pimgr.transform([dgm], skew=False)[0]

    return img.flatten()


# 4. Bağlı Bileşen ve Yoğunluk Hesaplama
def compute_metrics(positions, r_c):
    # Mesafe matrisi
    dist_matrix = distance.cdist(positions, positions)
    
    # Komşuluk matrisi (kendine bağlantı yok)
    adj_matrix = (dist_matrix <= r_c) & (dist_matrix > 0)
    
    # Grafik oluştur ve bağlı bileşenler
    G = nx.from_numpy_array(adj_matrix)
    beta0 = nx.number_connected_components(G)
    
    # Konveks zarf alanı (yoğunluk için)
    try:
        hull = ConvexHull(positions)
        area = hull.volume
    except:
        area = 100  # Minimum alan
    
    density = len(positions) / area
    return beta0, density

# 5. Adaptif İletişim Yarıçapı
def adaptive_radius(density, r_base, rho_thr, kappa):
    density_km2 = density * 1e6  # UAV/km²'ye dönüştür
    if density_km2 <= 25:  # ρ_thr = 25 UAV/km²
        return r_base
    else:
        return r_base * kappa

# 6. Ana Simülasyon Döngüsü
def run_simulation(config):
    dataset = []
    total_snapshots = len(config.N_list) * len(config.mobility_models) * config.num_runs * config.snapshots_per_run
    
    with tqdm(total=total_snapshots, desc="Generating Dataset") as pbar:
        for N in config.N_list:
            for model_type in config.mobility_models:
                for run in range(config.num_runs):
                    # Hareket modelini başlat
                    if model_type == 'RWP':
                        mobility = RandomWaypoint(config.area, N)
                    else:
                        mobility = GaussMarkov(config.area, N)
                    
                    positions = mobility.initialize()
                    
                    for t in range(config.snapshots_per_run):
                        # Pozisyonları güncelle
                        positions = mobility.step(config.dt)
                        
                        # Yoğunluğa göre adaptif yarıçap hesapla
                        _, density = compute_metrics(positions, config.r_base)
                        r_c_adaptive = adaptive_radius(
                            density, config.r_base, config.rho_thr, config.kappa)
                        
                        # Sabit ve adaptif bağlı bileşenleri hesapla
                        beta0_fixed, _ = compute_metrics(positions, config.r_base)
                        beta0_adaptive, _ = compute_metrics(positions, r_c_adaptive)
                        
                        # Topolojik özellikleri çıkar
                        persistence_image = compute_persistence_image(
                            positions, config.epsilon_max, config.pixels, config.sigma)
                        
                        # Veri noktasını kaydet
                        data_point = {
                            'positions': positions.copy(),
                            'beta0_fixed': beta0_fixed,
                            'beta0_adaptive': beta0_adaptive,
                            'persistence_image': persistence_image,
                            'r_c_adaptive': r_c_adaptive,
                            'N': N,
                            'model': model_type,
                            'run': run,
                            'time': t * config.dt
                        }
                        dataset.append(data_point)
                        pbar.update(1)
    
    return dataset

# 7. Veriyi HDF5 Formatında Kaydetme
def save_to_hdf5(dataset, filename):
    with h5py.File(filename, 'w') as hf:
        # Grup oluştur
        grp = hf.create_group('fanet_topo_dataset')
        
        # Veri setini kaydet
        for i, data in enumerate(dataset):
            subgroup = grp.create_group(f'snapshot_{i}')
            subgroup.create_dataset('positions', data=data['positions'])
            subgroup.create_dataset('persistence_image', data=data['persistence_image'])
            subgroup.attrs['beta0_fixed'] = data['beta0_fixed']
            subgroup.attrs['beta0_adaptive'] = data['beta0_adaptive']
            subgroup.attrs['r_c_adaptive'] = data['r_c_adaptive']
            subgroup.attrs['N'] = data['N']
            subgroup.attrs['model'] = data['model']
            subgroup.attrs['run'] = data['run']
            subgroup.attrs['time'] = data['time']

# 8. Ana İşlem
if __name__ == "__main__":
    config = SimulationConfig()
    print(f"Generating {config.num_runs * len(config.N_list) * len(config.mobility_models) * config.snapshots_per_run} snapshots...")
    
    dataset = run_simulation(config)
    save_to_hdf5(dataset, 'fanet_topo_dataset.h5')
    print("Dataset saved to fanet_topo_dataset.h5")