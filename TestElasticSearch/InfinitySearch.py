from __future__ import print_function
import os
import time
import glob
from tkinter import *
#import Tkinter, Tkconstants, tkFileDialog
from threading import Lock
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename 
from tkinter.filedialog import askdirectory 

from TestElasticSearch import NcsistSearchApi as InfinitySearchApi


class StopWatch():
    def __init__(self):
        # Get initial time in seconds since the Epoch.
        self.initial_time = time.time()
        self.last_mark_time = self.initial_time

    def mark(self, unit='sec'):
        """
        Make time span since initial
        :param unit:
        :return:
        """
        current_time = time.time()

        if unit == 'ms':
            time_span = (current_time - self.last_mark_time) * 1000
        else:
            time_span = current_time - self.last_mark_time

        self.last_mark_time = current_time
        return time_span

    def total_elapsed(self, unit='sec'):
        current_time = time.time()

        if unit == 'ms':
            time_span = (current_time - self.initial_time) * 1000
        else:
            time_span = current_time - self.initial_time
        return time_span

    def reset(self):
        self.initial_time = time.time()


class MainWindow(Frame):
    def __init__(self, master):
        sys.stdout = self  # redirect stdout to self write()/flush()
        self.master = master
        self.frame = Frame(master)
        self.frame.pack()
        self.master.title('Infinity Search 1.0')
        self.lock_output = Lock()

        # GUI variables
        self.json_folder = StringVar()
        self.json_folder.set(r'/data1/JC_Sample/sample_data_only_face')
        self.query_file = StringVar()
        self.query_file.set(r'/data1/JC_Sample/query1.json')
        self.es_server = StringVar()
        self.es_server.set('hosts=10.10.53.201,10.10.53.204,10.10.53.207;port=9200;id=esadmin;passwd=esadmin@2018')
        self.search_field = StringVar()
        self.search_field.set('imgPath')
        self.search_text = StringVar()

        self.image_no = Image.open('no-image.png')
        self.image_no.thumbnail((100, 100), Image.ANTIALIAS)
        self.label_list_file = []
        self.label_list_image = []
        self.label_list_photo = []
        self.label_list_score = []
        self.image_path = r'/data1/JC_Sample/sample_data_only_face'

        self.update_id = StringVar()
        self.update_id.set('LMTSvmYBxjzL5lGNJfi-')
        self.update_field = StringVar()
        self.update_field.set('category')
        self.update_text = StringVar()
        self.update_text.set('face,wang')

        # ES variable
        self.es = None

        self.setup_widget()

    def setup_widget(self):
        # Panel layout
        Grid.rowconfigure(self.master, 0, weight=1)
        Grid.columnconfigure(self.master, 0, weight=1)
        self.frame.grid(row=0, column=0, sticky='NEWS')

        Grid.rowconfigure(self.frame, 0, weight=0)
        Grid.rowconfigure(self.frame, 1, weight=0)
        Grid.rowconfigure(self.frame, 2, weight=1)

        self.setup_widget_operation()
        self.setup_widget_result()
        self.setup_widget_output()

    def setup_widget_operation(self):
        optFrame = Frame(self.frame, borderwidth=1, relief="sunken")
        Grid.rowconfigure(optFrame, 0, weight=0)
        Grid.columnconfigure(optFrame, 0, weight=0)
        Grid.columnconfigure(optFrame, 1, weight=4)
        Grid.columnconfigure(optFrame, 2, weight=4)
        Grid.columnconfigure(optFrame, 3, weight=0)

        # ES Server
        Label(optFrame, text='ES Conn Str:') \
            .grid(row=0, column=0, sticky=(E), padx=3, pady=3)
        Entry(optFrame, width=60, textvariable=self.es_server) \
            .grid(row=0, column=1, columnspan=2, sticky='NEWS', padx=3, pady=3)
        Button(optFrame, text="Connect", command=self.cmd_connect_es) \
            .grid(row=0, column=3, sticky='NEWS', padx=3, pady=3)
        # Import JSON selection
        Button(optFrame, text="Load:", command=self.cmd_select_folder) \
            .grid(row=1, column=0, sticky='NEWS', padx=3, pady=3)
        Entry(optFrame, width=60, textvariable=self.json_folder, state=DISABLED) \
            .grid(row=1, column=1, columnspan=2, sticky='NEWS', padx=3, pady=3)
        Button(optFrame, text="Process", command=self.cmd_load) \
            .grid(row=1, column=3, sticky='NEWS', padx=3, pady=3)
        # Query
        Button(optFrame, text="Query:", command=self.cmd_select_file) \
            .grid(row=2, column=0, sticky='NEWS', padx=3, pady=3)
        Entry(optFrame, width=60, textvariable=self.query_file, state=DISABLED) \
            .grid(row=2, column=1, columnspan=2, sticky='NEWS', padx=3, pady=3)
        Button(optFrame, text="Process", command=self.cmd_query) \
            .grid(row=2, column=3, sticky='NEWS', padx=3, pady=3)
        # Search
        Label(optFrame, text='Search:') \
            .grid(row=3, column=0, sticky=(E), padx=3, pady=3)
        Entry(optFrame, width=20, textvariable=self.search_field) \
            .grid(row=3, column=1, sticky='NEWS', padx=3, pady=3)
        Entry(optFrame, width=40, textvariable=self.search_text) \
            .grid(row=3, column=2, sticky='NEWS', padx=3, pady=3)
        Button(optFrame, text="Process", command=self.cmd_search) \
            .grid(row=3, column=3, sticky='NEWS', padx=3, pady=3)
        # Update
        Label(optFrame, text='Update:') \
            .grid(row=4, column=0, sticky=(E), padx=3, pady=3)
        Entry(optFrame, width=20, textvariable=self.update_id) \
            .grid(row=4, column=1, sticky='NEWS', padx=3, pady=3)
        Entry(optFrame, width=40, textvariable=self.update_field) \
            .grid(row=4, column=2, sticky='NEWS', padx=3, pady=3)
        Button(optFrame, text="Process", command=self.cmd_update) \
            .grid(row=4, column=3, sticky='NEWS', padx=3, pady=3)

        Entry(optFrame, width=40, textvariable=self.update_text) \
            .grid(row=5, column=2, sticky='NEWS', padx=3, pady=3)

        # Clear All document
        commandFrame = Frame(optFrame, borderwidth=1)
        Button(commandFrame, text="Clear All", command=self.cmd_clear_all_doc) \
            .grid(row=0, column=0, sticky='NEWS', padx=3, pady=3)
        Button(commandFrame, text="Count", command=self.cmd_count) \
            .grid(row=0, column=1, sticky='NEWS', padx=3, pady=3)
        commandFrame \
            .grid(row=6, column=0, columnspan=4, sticky='NEWS', padx=3, pady=3)

        optFrame \
            .grid(row=0, column=0, sticky='NEWS', padx=3, pady=3)

    def setup_widget_result(self):
        # Query result panel
        picFrame = Frame(self.frame, borderwidth=1, relief="sunken")
        Grid.rowconfigure(picFrame, 0, weight=0)
        Grid.rowconfigure(picFrame, 1, weight=1)
        Grid.rowconfigure(picFrame, 2, weight=0)
        Grid.columnconfigure(picFrame, 0, weight=1)
        Grid.columnconfigure(picFrame, 1, weight=1)
        Grid.columnconfigure(picFrame, 2, weight=1)
        Grid.columnconfigure(picFrame, 3, weight=1)
        Grid.columnconfigure(picFrame, 4, weight=1)

        for i in range(5):
            label = Label(picFrame, text='file')
            label.grid(row=0, column=i, sticky='EW', padx=3, pady=3)
            self.label_list_file.append(label)

            photo = ImageTk.PhotoImage(self.image_no)
            self.label_list_photo.append(photo)

            label = Label(picFrame, image=photo)
            label.grid(row=1, column=i, sticky='EW', padx=3, pady=3)
            self.label_list_image.append(label)

            label = Label(picFrame, text='0.0')
            label.grid(row=2, column=i, sticky='EW', padx=3, pady=3)
            self.label_list_score.append(label)

        picFrame \
            .grid(row=1, column=0, sticky='NEWS', padx=3, pady=3)

    def setup_widget_output(self):
        # Message output panel
        txtFrame = Frame(self.frame, borderwidth=1, relief="sunken")
        self.txt_output = Text(txtFrame, wrap=NONE, font='arial 9', height=16, borderwidth=0)
        vscroll = Scrollbar(txtFrame, orient=VERTICAL, command=self.txt_output.yview)
        self.txt_output['yscroll'] = vscroll.set
        vscroll.pack(side="right", fill="y")
        self.txt_output.pack(side="left", fill="both", expand=True)
        txtFrame \
            .grid(row=2, column=0, sticky='NEWS', padx=3, pady=3)
        self.txt_output.insert(INSERT, 'Program started\n')

    def write(self, text):
        self.lock_output.acquire()
        try:
            self.txt_output.insert(INSERT, text)
            self.txt_output.see(END)
        finally:
            self.lock_output.release()

    def flush(self):
        pass

    def nonblock_stdout(self, out, thd_id = 0):
        for line in iter(out.readline, b''):
            line = line.decode('cp950')
            if len(line) > 4:
                print('\t({0}): {1}'.format(thd_id, line), end='')
        out.close()

    def nonblock_stderr(self, out, thd_id = 0):
        for line in iter(out.readline, b''):
            line = line.decode('cp950')
            if len(line) > 4:
                print('\t({0}): {1}'.format(thd_id, line), end='')
        out.close()

    def clear_output(self):
        self.txt_output.delete('1.0', END)

    def display_result(self, pic_list):
        if pic_list:
            for i in range(5):
                img_file = os.path.join(self.image_path, pic_list[i]['_source']['imgPath'])
                dir_name = os.path.basename(os.path.dirname(img_file))
                self.label_list_file[i].configure(text=dir_name)
                image = Image.open(img_file)
                image.thumbnail((100, 100), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(image)
                self.label_list_photo[i] = photo
                self.label_list_image[i].configure(image=self.label_list_photo[i])
                score = '{0:7.6f}'.format(pic_list[i]['_score'])
                self.label_list_score[i].configure(text=score)
        else:
            for i in range(5):
                self.label_list_file[i].configure(text='file')
                self.label_list_photo[i] = ImageTk.PhotoImage(self.image_no)
                self.label_list_image[i].configure(image=self.label_list_photo[i])
                self.label_list_score[i].configure(text='0.0')
            pass

    def cmd_connect_es(self):
        try:
            stopwatch = StopWatch()
            self.clear_output()
            self.es = InfinitySearchApi.InfinitySearch(self.es_server.get())
            status = self.es.status()
            print('Elapsed: {0:6.2f} sec(s)'.format(stopwatch.mark()))
            print(status)
        except Exception as ex:
            print(ex)

    def cmd_select_folder(self):
        folder = askdirectory()
        if folder:
            self.json_folder.set(folder)

    def cmd_select_file(self):
        file = askopenfilename(title="Select file", filetypes=(("json files", "*.json"), ("all files", "*.*")))
        if file:
            self.query_file.set(file)

    def cmd_load(self):
        folder = self.json_folder.get()
        stopwatch = StopWatch()
        if folder and self.es:
            files = glob.glob(os.path.join(folder, '*.json'))
            files.sort()
            for f in files:
                self.es.load_data(f)
        print('Elapsed: {0:6.2f} sec(s)'.format(stopwatch.mark()))

    def cmd_count(self):
        result = self.es.count()
        print(result)

    def cmd_clear_all_doc(self):
        if self.es:
            status = self.es.delete_all()
            print(status)
        pass

    def cmd_query(self):
        self.clear_output()
        self.display_result(None)
        file = self.query_file.get()
        stopwatch = StopWatch()
        if file and self.es:
            result = self.es.query_result(file)
            self.display_result(result)
        print('Elapsed: {0:6.2f} sec(s)'.format(stopwatch.mark()))

    def cmd_search(self):
        self.clear_output()
        self.display_result(None)
        field = self.search_field.get()
        search_text = self.search_text.get()
        stopwatch = StopWatch()
        if self.es and field.__len__() > 0 and search_text.__len__() > 0:
            result = self.es.search_result(field, search_text)
            self.display_result(result)
        print('Elapsed: {0:6.2f} sec(s)'.format(stopwatch.mark()))

    def cmd_update(self):
        self.clear_output()
        self.display_result(None)
        id_no = self.update_id.get()
        field = self.update_field.get()
        text = self.update_text.get()
        stopwatch = StopWatch()
        if self.es and len(id_no) > 0 and len(field) > 0 and len(text) > 0:
            result = self.es.update_fields(id_no, field, text)
        print('Elapsed: {0:6.2f} sec(s)'.format(stopwatch.mark()))


def main():
    root = Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == '__main__':
    main()
