import os
import common
import argparse
import numpy as np
import trimesh
from tqdm import tqdm

class Scale:
    """
    Scales a bunch of meshes.
    """

    def __init__(self):
        """
        Constructor.
        """

        parser = self.get_parser()
        self.options = parser.parse_args()

    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """

        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as PLY files.')
        parser.add_argument('--in_dir', type=str, help='Path to input directory.')
        parser.add_argument('--out_dir', type=str, help='Path to output directory; files within are overwritten!')
        parser.add_argument('--padding', type=float, default=0.1, help='Relative padding applied on each side.')
        return parser

    def read_directory(self, directory):
        """
        Read directory.

        :param directory: path to directory
        :return: list of files
        """

        files = []
        for filename in os.listdir(directory):
            files.append(os.path.normpath(os.path.join(directory, filename)))

        return files

    def run(self):
        """
        Run the tool, i.e. scale all found PLY files.
        """

        assert os.path.exists(self.options.in_dir)
        assert os.path.exists(os.path.join(self.options.in_dir, 'tooth_crown'))
        assert os.path.exists(os.path.join(self.options.in_dir, 'jaw'))
         
        common.makedir(self.options.out_dir)
        common.makedir(os.path.join(self.options.out_dir, 'tooth_crown'))
        out_jaw_path = os.path.join(self.options.out_dir, 'jaw')
        common.makedir(out_jaw_path)
        files = self.read_directory(os.path.join(self.options.in_dir, 'tooth_crown'))

        for filepath in tqdm(files, desc='Processing scaling: ', total=len(files)):
            if filepath[-4:] != '.ply':
                continue
            mesh = trimesh.load(filepath)
            mesh_upper_jaw = trimesh.load(os.path.join(self.options.in_dir, "jaw", "upper_" + filepath.split('/')[-1]))
            mesh_lower_jaw = trimesh.load(os.path.join(self.options.in_dir, "jaw", "lower_" + filepath.split('/')[-1]))


            # Get extents of model.
            min = mesh.vertices.min(axis=0)
            max = mesh.vertices.max(axis=0)


            # Set the center (although this should usually be the origin already).
            centers = (
                (min[0] + max[0]) / 2,
                (min[1] + max[1]) / 2,
                (min[2] + max[2]) / 2
            )
            

            sizes = max - min
            logest_size = sizes.max()

            translation = (
                -centers[0],
                -centers[1],
                -centers[2]
            )


            scales = (
                1 / (logest_size + self.options.padding * 2 * logest_size)
            )
            

            mesh.vertices += translation
            mesh.vertices *= scales

            mesh_upper_jaw.vertices += translation
            mesh_upper_jaw.vertices *= scales

            mesh_lower_jaw.vertices += translation
            mesh_lower_jaw.vertices *= scales

            print('[Data] %s extents before %f - %f, %f - %f, %f - %f' % (os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))
            min = mesh.vertices.min(axis=0)
            max = mesh.vertices.max(axis=0)
            print('[Data] %s extents after %f - %f, %f - %f, %f - %f' % (os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))

            # May also switch axes if necessary.
            # mesh.vertices[:, [0, 2]] = mesh.vertices[:, [2, 0]]
            # mesh_upper_jaw.vertices[:, [0, 2]] = mesh_upper_jaw.vertices[:, [2, 0]]
            # mesh_lower_jaw.vertices[:, [0, 2]] = mesh_lower_jaw.vertices[:, [2, 0]]

            output_filepath = os.path.join(self.options.out_dir, "tooth_crown", os.path.splitext(os.path.basename(filepath))[0] + '.off')
            mesh.export(output_filepath)
            output_upper_path = os.path.join(out_jaw_path, "upper_" + os.path.splitext(os.path.basename(filepath))[0] + '.off')
            output_lower_path = os.path.join(out_jaw_path, "lower_" + os.path.splitext(os.path.basename(filepath))[0] + '.off')
            mesh_upper_jaw.export(output_upper_path)
            mesh_lower_jaw.export(output_lower_path)

if __name__ == '__main__':
    app = Scale()
    app.run()