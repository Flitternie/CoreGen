import subprocess


class Bleu:
    def __init__(self):
        pass

    def compute_score(self, predicted_file_name, ref_file_name):
        with open(predicted_file_name) as predicted_file:
            pipe = subprocess.Popen(["perl", "/home/shangqing/Documents/GitHub/code2commit/evaluation/multi-bleu.perl", ref_file_name], stdin=predicted_file,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = pipe.communicate()
            return stdout, stderr
