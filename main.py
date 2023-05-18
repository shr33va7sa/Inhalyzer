import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, decimate
from kivy_garden.graph import Graph, MeshLinePlot
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView

def sampenc(y, M, r):
    n = len(y)
    lastrun = np.zeros(n, dtype=int)
    run = np.zeros(n, dtype=int)
    A = np.zeros(1, dtype=int)
    B = np.zeros(1, dtype=int)
    p = np.zeros(1)
    e = np.zeros(1)
    for i in range(n-1):
        nj = n-i
        y1 = y[i]
        for jj in range(nj):
            j = jj+i
            if abs(y[j]-y1) < r:
                run[jj] = lastrun[jj]+1
                M1 = min(1, run[jj])
                for m in range(M1):
                    A[m] += 1
                    if j < n-1:
                        B[m] += 1
            else:
                run[jj] = 0
        lastrun[:nj] = run[:nj]
    N = n*(n-1)//2
    p[0] = A[0]/N
    e[0] = -np.log(p[0])
    for m in range(1, 1):
        p[m] = A[m]/B[m-1]
        e[m] = -np.log(p[m])
    return e, A, B

class GraphApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        # Create a FileChooser widget
        file_chooser = FileChooserListView(path='/', filters=['*.wav'])

        # Create a button to load the selected file
        load_button = Button(text='Load File', size_hint=(None, None), size=(200, 50))

        def load_file(instance):
            # Get the selected file path
            selected_file = file_chooser.selection and file_chooser.selection[0]
            if selected_file:
                fs1, x = wavfile.read(selected_file)

                # Differentiate the signal
                s = np.diff(x)

                # Calculate envelope
                h = signal.hilbert(s)
                A = np.abs(h)
                filtered1 = signal.filtfilt(*signal.butter(2, 2 / fs1, 'low'), A)
                filtered1 /= np.abs(filtered1).max()

                # Calculate minima points
                F = np.gradient(filtered1)
                s_3, = np.where((F[:-1] <= 0) & (F[1:] > 0))

                # Calculate the indices of the minima-to-minima cycle
                cycle_start = s_3[3]
                cycle_end = s_3[5]

                # Calculate e1 value
                NFFT = 2048
                Y = np.fft.fft(x[10000:120000], NFFT)
                a = abs(Y[:NFFT // 2])

                c2 = np.std(a[1:500])
                r1 = 0.3 * c2
                M1 = 1
                e1, A1, B1 = sampenc(a[:500], M1, r1)

                # Extract the minima-to-minima cycle data
                cycle_data = x[cycle_start:cycle_end + 1]
                # Calculate the number of samples in cycle_data
                num_samples = len(cycle_data)
                breathrate = (60 * fs1) / num_samples
                breathcycle= 60/breathrate

                # Print the number of samples in cycle_data
                print(f"Breath Rate: {breathrate}")
                print(f"Average Time Of a Breath: {breathcycle}")

                # Downsample the cycle data
                downsample_factor = 10  # Adjust this factor as needed
                cycle_data_downsampled = decimate(cycle_data, downsample_factor)

                # Downsample the x data
                x_downsampled = decimate(x, downsample_factor)

                # Create the graph objects
                graph1 = Graph(xlabel='Sample number', ylabel='Amplitude')
                graph2 = Graph(xlabel='Sample number', ylabel='Amplitude')

                # Create the x plot
                x_plot = MeshLinePlot(color=[1, 0, 0, 1])
                x_plot.points = [(i, x_downsampled[i]) for i in range(len(x_downsampled))]

                # Create the cycle_data plot
                cycle_data_plot = MeshLinePlot(color=[0, 1, 0, 1])
                cycle_data_plot.points = [(i, cycle_data[i]) for i in range(len(cycle_data))]

                # Add the plots to their respective graphs
                graph1.add_plot(x_plot)
                graph2.add_plot(cycle_data_plot)

                # Adjust the graph views to show the entire plotted data
                graph1.xmax = str(len(x_downsampled) - 1)
                graph1.ymin = str(min(x_downsampled))
                graph1.ymax = str(max(x_downsampled))
                graph2.xmax = str(len(cycle_data) - 1)
                graph2.ymin = str(min(cycle_data))
                graph2.ymax = str(max(cycle_data))

                # Create the app layout
                #layout = BoxLayout(orientation='vertical')
                layout.clear_widgets()
                if e1 < 1:
                    diagnosis = "Normal Functioning"
                else:
                    diagnosis = "Abnormal Functioning"
                # label = Label(text={e1}diagnosis, font_size='20sp')
                label = Label(text=f'Breathing Rate:  {round(breathrate)} Breaths/sec', font_size='30sp')
                label1 = Label(text=f'Average Time of One Breath:  {breathcycle} seconds', font_size='30sp')
                label2 = Label(text=f'Lung Status:  {diagnosis}', font_size='30sp')
                # Add

                # Add the graphs to the layout
                layout.add_widget(graph1)
                layout.add_widget(graph2)
                layout.add_widget(label)
                layout.add_widget(label1)
                layout.add_widget(label2)

        load_button.bind(on_release=load_file)
        layout.add_widget(file_chooser)
        layout.add_widget(load_button)


        return layout

if __name__ == '__main__':
    GraphApp().run()