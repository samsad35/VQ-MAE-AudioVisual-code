import torch
print(torch.cuda.is_available())
from src import InterfaceVQMAE


if __name__ == '__main__':
    gui_vqmae = InterfaceVQMAE()
    gui_vqmae.master.mainloop()
