import os

with open("Open_Scientific_Visualization_Datasets.html", "r") as file:
    lines = file.readlines()

i = 0
for line in lines:
    line = line.strip()
    if 'class="download"' in line:
        start = line.find('href="') + 6
        end = line.find('.raw"', start) + len('.raw"') - 1
        url = line[start:end]
        i += 1
        assert url.startswith("http"), f"URL {url} does not start with 'http'"
        filename = url.split("/")[-1]
        if not os.path.exists(filename):
            print(f"Downloading ({i}) {filename} from {url}")
            os.system(f"wget {url}")
        else:
            print(f"{filename} already exists, skipping download.")
