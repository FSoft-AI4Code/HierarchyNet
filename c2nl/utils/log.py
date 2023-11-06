import pandas as pd
import matplotlib.pyplot as plt

class Log:
    @staticmethod
    def write_header(log_file_path, header):
        with open(log_file_path, 'w') as f:
            f.write(header + '\n')
    @staticmethod
    def write_log(log_file_path, data):
        with open(log_file_path, 'a') as f:
            f.write(data + '\n')
    @staticmethod
    def visualize_logs(logs_path, figure_path):
        df = pd.read_csv(logs_path)
        columns = df.columns[1:]
        epochs = df[df.columns[0]].values
        plt.figure(figsize = (12, 12))
        plt.style.use('ggplot')
        for col in columns:
            plt.plot(epochs, df[col].values, label = col)
        plt.xticks(epochs, epochs, rotation = 30)
        plt.xlabel('Epoch #')
        plt.title('Training process' if 'train' in logs_path else 'Validation process')
        plt.legend()
        plt.savefig(figure_path)
        plt.close()
