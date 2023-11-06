import os
import shutil
import datetime
from ftplib import FTP
import subprocess

class FTPTransfer:
    def __init__(self, ftp_credentials):
        self.ftp_credentials = ftp_credentials
        self.ftp = FTP()
        self.dir = None

    def connect(self):
        self.ftp.connect(self.ftp_credentials['ftp_host'], self.ftp_credentials['ftp_port'])
        self.ftp.login(user=self.ftp_credentials['ftp_user'], passwd=self.ftp_credentials['ftp_pass'])
        print('Connected to FTP ...')

    def upload_file(self, file_path):
        curr_datetime = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        remote_file_path = self.ftp_credentials['permision_path'] + f'/{["Cam_name"]}-{curr_datetime}.png'
        
        try:
            self.ftp.cwd(self.ftp_credentials['permision_path'])
            print('Changing FTP Directory ...')
            self.ftp.storbinary(f"STOR {remote_file_path}", open(file_path, "rb"))
            print('Data has been stored ...')
        except Exception as e:
            print('There is a problem in storing the data:', str(e))
            pass

    def disconnect(self):
        self.ftp.quit()


class USBFileManager:
    def __init__(self):
        self.debug_mode = None
        self.usb_drives = []

    def get_free_space(self, path):
        # Return the available space in bytes on the given path
        stat = os.statvfs(path)
        return stat.f_frsize * stat.f_bavail
    
    def check_usb(self):
        # Search for available USB drives
        for drive in os.listdir("/dev"):
            if drive.startswith("sd") and drive.endswith("1"): # or drive.startswith("mmcblk"):
                drive_path = os.path.join("/dev", drive)
                self.usb_drives.append(drive_path)

        # Display available USB drives
        if self.usb_drives:
            print("USB drive(s) detected:")
            for drive in self.usb_drives:
                print(drive)
            print("Use the 'transfer_file' function to transfer files.")
            self.debug_mode = True
        else:
            print("USB drive(s) not found.")
            self.debug_mode = True

    def mount_drive(self, drive_path, mount_path):
        # Check if the mount path exists, if not, create it
        if not os.path.exists(mount_path):
            subprocess.run(["sudo","mkdir",mount_path])
            # os.makedirs(mount_path)

        # Mount the USB drive
        os.system(f"sudo mount {drive_path} {mount_path}")
        print(f"USB drive mounted at {mount_path}")

    def format_drive(self, drive_path, fs_type="vfat"):
        # Format the USB drive with the specified filesystem type (default is VFAT)
        print(f"Formatting {drive_path} with {fs_type} filesystem. This will erase all data on the drive.")
        subprocess.run(["sudo", "mkfs", "-t", fs_type, drive_path])
        print(f"{drive_path} formatted successfully.")

    def format_usb(self, fs_type="vfat"):
        # Format all available USB drives
        if self.usb_drives:
            for drive in self.usb_drives:
                self.format_drive(drive, fs_type)
        else:
            print("USB drive(s) not found. Unable to format.")

    def transfer_file(self, source_path, destination_path):
        # Check if any USB drives are available
        if self.usb_drives:
            # Mount each USB drive and transfer the file
            for drive in self.usb_drives:
                mount_path = f"/media/usb_{drive.split('/')[-1]}"
                self.mount_drive(drive, mount_path)
                mount_path = os.path.join(mount_path, self.dir)
                try:
                    # Check available space on the USB drive
                    free_space = self.get_free_space(mount_path)
                    file_size = os.path.getsize(source_path)
                    if file_size > free_space:
                        print(f"Not enough space on {drive}. Skipping...")
                        continue
                    subprocess.run(["sudo", "cp","-rf", source_path, os.path.join(mount_path, destination_path)])
                    # shutil.copy2(source_path, os.path.join(mount_path, destination_path))
                    print(f"File transferred to {drive} successfully.")
                except IOError as e:
                    print(f"Error transferring file to {drive}: {e}")
                    

                # Unmount the USB drive
                os.system(f"sudo umount {mount_path}")
                print(f"USB drive at {mount_path} dismounted.")
        else:
            print("USB drive(s) not found. Unable to transfer file.")
    
    def make_directory_on_usb(self, directory_name):
        # Check if any USB drives are available
        if self.usb_drives:
            # Create the directory on each USB drive
            for drive in self.usb_drives:
                mount_path = f"/media/usb_{drive.split('/')[-1]}"
                self.mount_drive(drive, mount_path)

                try:
                    new_directory = os.path.join(mount_path, directory_name)
                    subprocess.run(["sudo","mkdir",new_directory])
                    # os.makedirs(new_directory)
                    self.dir = new_directory
                    print(f"Directory '{directory_name}' created on {drive}.")
                except OSError as e:
                    print(f"Error creating directory on {drive}: {e}")

                # Unmount the USB drive
                os.system(f"sudo umount {mount_path}")
                print(f"USB drive at {mount_path} dismounted.")
        else:
            print("USB drive(s) not found. Unable to create directory.")



