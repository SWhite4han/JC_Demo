from TestElasticSearch import NcsistSearch
from tkinter import Tk


def main():
    root = Tk()
    NcsistSearch.MainWindow(root)
    root.mainloop()


if __name__ == '__main__':
    main()
