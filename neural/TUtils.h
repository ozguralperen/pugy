#include "Tensor.hh"
#include "Bitmap.hpp"
#include <string>
#include <iostream>
#include <vector>

void TSaveToBitmap(Tensor &tensor , const std::string &path, int blockSize){
	int width = tensor.size.Width * blockSize;
	int height = tensor.size.Height * blockSize;

	if (tensor.size.Width == 1 && tensor.size.Height == 1) {
		BitmapImage image(blockSize, tensor.size.Deep * blockSize);

		for (int d = 0; d < tensor.size.Deep; d++) {
			int value = std::min(255.0, std::fmax(0.0, tensor.Data[d]));

			for (int i = 0; i < blockSize; i++)
				for (int j = 0; j < blockSize; j++)
					image.set_pixel(j, d * blockSize + i, 0, value, 0);
		}
		
		image.save_image(path + ".bmp");
	}
	else if (tensor.size.Deep == 3) {
		BitmapImage image(width, height);

		for (int y = 0; y < tensor.size.Height; y++) {
			for (int x = 0; x < tensor.size.Height; x++) {
				int r = std::min(255.0, std::fmax(0.0, tensor(0, y, x)));
				int g = std::min(255.0, std::fmax(0.0, tensor(1, y, x)));
				int b = std::min(255.0, std::fmax(0.0, tensor(2, y, x)));

				for (int i = 0; i < blockSize; i++)
					for (int j = 0; j < blockSize; j++)
						image.set_pixel(x * blockSize + j, y * blockSize + i, r, g, b);
			}
		}
		
		image.save_image(path + ".bmp");
	}
	else {
		for (int d = 0; d < tensor.size.Height; d++) {
			BitmapImage image(width, height);

			for (int y = 0; y < tensor.size.Height; y++) {
				for (int x = 0; x < tensor.size.Width; x++) {
					int br = std::min(255.0, std::fmax(0.0, tensor(d, y, x)));
					
					for (int i = 0; i < blockSize; i++)
						for (int j = 0; j < blockSize; j++)
							image.set_pixel(x * blockSize + j, y * blockSize + i, br, br, br);
				}
			}
			
			if (tensor.size.Deep == 1) {
				image.save_image(path + ".bmp");
			}
			else {
				image.save_image(path + "_ch" + std::to_string(d) + ".bmp");
			}
		}
	}
}