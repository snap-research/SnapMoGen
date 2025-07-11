rm -rf checkpoint_dir
mkdir checkpoint_dir
cd checkpoint_dir
mkdir humanml3d

cd humanml3d 
echo -e "Downloading pretrained models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1NyL8S7_88x_LBBQILg2tQ_V2W4aJQ4Gv/view?usp=sharing

echo -e "Unzipping humanml3d_models.zip"
unzip humanml3d_models.zip

echo -e "Cleaning humanml3d_models.zip"
rm humanml3d_models.zip

cd ../
mkdir snapmogen
cd snapmogen

echo -e "Downloading pretrained models for SnapMoGen dataset"
gdown --fuzzy https://drive.google.com/file/d/19Uzj51v5BU98EeaEgmpHjnpYab2gQMMJ/view?usp=sharing

echo -e "Unzipping snapmogen_models.zip"
unzip snapmogen_models.zip

echo -e "Cleaning snapmogen_models.zip"
rm snapmogen_models.zip

cd ../../

echo -e "Downloading done!"