# Create .csv file out of an image folder

import argparse
import csv
import glob
import os.path
import os.path as path


def main(src_path, csv_path, out_path, discarded_out_path):
	if not path.exists(src_path):
		raise ValueError(f"Source folder \"{src_path}\" does not exist")
	if not path.exists(csv_path):
		raise ValueError(f"Csv file \"{csv_path}\" does not exist")
	if path.exists(out_path):
		raise ValueError("CSV Output file already exists")
	if path.exists(discarded_out_path):
		raise ValueError("Discarded CSV Output file already exists")

	files = glob.glob(f"{src_path}/*.jpg")

	files = [os.path.splitext(os.path.basename(file))[0] for file in files]

	# Important to set universal newline to blank to avoid CSV module from inserting its own carriage return (especially on windows)
	with open(csv_path, 'r') as source_file, \
			open(out_path, 'w', newline='') as out_file, \
			open(discarded_out_path, 'w', newline='') as discarded_file:

		source_file_reader = csv.reader(source_file)
		out_file_writer = csv.writer(out_file)
		discarded_out_writer = csv.writer(discarded_file)

		end = len(files)
		i = 0
		needle = files[i]

		for row in source_file_reader:
			if os.path.splitext(os.path.basename(row[3]))[0] == needle and i < end:
				out_file_writer.writerow(row)

				i += 1
				if i < end:
					needle = files[i]

				print(f"Found: {row}")
			else:
				discarded_out_writer.writerow(row)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('source_folder')
	parser.add_argument('source_csv')
	parser.add_argument('out_path')
	parser.add_argument('discarded_path')

	args = parser.parse_args()

	print(f"Source folder: {args.source_folder}")
	print(f"Source CSV: {args.source_csv}")
	print(f"CSV Output: {args.out_path}")
	print(f"Discarded rows CSV: {args.discarded_path}")

	print("Filtering..")

	main(args.source_folder, args.source_csv, args.out_path, args.discarded_path)

	print("Finished.")
