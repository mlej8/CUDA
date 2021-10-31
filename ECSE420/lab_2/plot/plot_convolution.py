import matplotlib.pyplot as plt
import os

img_1 = [27555535 ,8920751 ,4099062 ,1653385 ,531011     ,478914     ,483267     ,513091     ,664804]
img_2=  [7032233 ,1485577 ,744612   ,374755   ,122049   ,117248   ,109409   ,113185   ,152193]
img_3 = [7412587,1650473,826916,419651,134625,121472,121633,123649,169249]
num_threads = [1, 4, 8, 16, 64, 128, 256, 512, 1024]
dirname = os.path.dirname(os.path.abspath(__file__))
for i, runtimes in enumerate([img_1, img_2, img_3], start=1):
    plt.plot(num_threads, runtimes)
    plt.xlabel("Block size")
    plt.ylabel("Convolution Kernel Runtime (ns)")
    plt.title(f"Convolution kernel runtime vs. block size for image {i}")
    plt.savefig(os.path.join(dirname,f"./image_{i}_runtime.png"))
    plt.clf()

    speedups = [runtimes[0] / r  for r in runtimes]
    print(f"Speed ups for image {i}: ", speedups)
    plt.plot(num_threads, speedups)
    plt.xlabel("Block size")
    plt.ylabel("Speedup")
    plt.title(f"Speedup of convolution kernel vs. block size for image {i}")
    plt.savefig(os.path.join(dirname, f"image_{i}_speedup.png"))
    plt.clf()
