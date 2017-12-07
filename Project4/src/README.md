# Getting Setup
### Building OpenCV with Extra Modules

##### 1. Download file and install necessary software
First you need to install Visual Studio. VS2013 has consistently worked.  VS Community 2015 works fine (despite compilation error in OpenCV modules). You will also need to download and install [CMake for Windows](https://cmake.org/download/).

Finally, download OpenCV and the extra module zips. [OpenCV 3.0.0](https://github.com/Itseez/opencv/tree/3.0.0) and [OpenCV Contrib 3.0.0](https://github.com/Itseez/opencv_contrib/tree/3.0.0). **Ensure you are on the 3.0.0 tag. The master branch is still at 2.4.** Unzip both files and put them on your desktop for now.

##### 2. Making Visual Studio Project with CMake
We need a destination for the OpenCV build. It's recommended that you create a directory in your root drive called C:\OpenCV.

Open up CMake and point it to the OpenCV source code directory (the files that you downloaded), and set it to build in the OpenCV folder.

![image](http://i.imgur.com/4TkH7Wo.png)

Next click the "Configure" button and choose Visual Studio XX 20XX Win64 (whichever version you installed before). You'll also want to select "Specify native compilers".

![image](http://i.imgur.com/CUfy9hW.png)
w
Both the C and the C++ compilers should be set to C:/Program Files (x86)/YOUR\_VISUAL\_STUDIO_VERSION/VC/bin/cl.exe. Hit "Finish" when you're done and let it run it's initial configuration. This may take a little while.

![image](http://i.imgur.com/UCstzPI.png)

When it's complete you should see this:

![image](http://i.imgur.com/riRvCAU.png)

Expand OPENCV and find OPENCV_EXTRA_MODULES_PATH. Point it to the opencv_contrib-3.0.0/modules directory that should be on your desktop. Once you've set the path, hit the "Configure" button one last time.

![image](http://i.imgur.com/kZex0fl.png)

If you expand BUILD when it's complete, you should be able to see all of the extra modules that will be built. Now click "Generate" and it should generate the build files in C:\OpenCV\

![image](http://i.imgur.com/Wt8qMUX.png)

##### 3. Building OpenCV

We'll now use Visual Studio to build OpenCV. In C:\OpenCV\ there should be a Visual Studio project called INSTALL.vcxproj. Open this in VS and let it process all of the files (this also may take a while). Once it's complete, look in the Solution Explorer for CMakeTargets -> INSTALL. Right click on INSTALL and choose "Build". This process will also take some time. You can continue with "Setting up the Environment Variables" and "Git (Version Control)" while it builds.

![image](http://i.imgur.com/YyUoBB0.png)

## Setting up the Environment Variables

Next we need to add a new variable (OPENCV_DIR) and update the system path variable. Locate the Environmental Variables configuration for your version of Windows. Once there, we'll first add a new variable called OPENCV_DIR and set it's path as C:\OpenCV\install\x64\vc12 (if you are using 2013). The vcXX will depend on which version of visual studio you are using.

![image](http://i.imgur.com/joo5TKk.png)

After you've created the OPENCV_DIR variable, we'll need to update the Path variable. In the System variables table, find "Path", and choose "Edit...". At the end of the existing variable, append ";%OPENCV_DIR%\bin". The result should look like this (NOTE: your existing path variable may be different, but the ending should be the same):

![image](http://i.imgur.com/h18Mnnj.png)

## Git (Version Control)

##### 1. Configuration

We are using Git as our version control system, so you'll need to do some configuration to get setup. You can skip this if you already have it configured on your machine.

Start by [downloading the Git command line tool for windows](https://git-for-windows.github.io/). Run the installer and when you get to "Adjusting your PATH environment" choose "Use Git from the Windows Command Prompt". For line ending conversions, leave it as the default: "Checkout Windows-style, commit Unix-style line endings". You should use MinTTY. Git should now be available from the Command Prompt. You can verify this by opening Command Prompt and running:
> git --version

It should return the version of git you are running if git was successfully installed.
##### 2. Learning Git

Understanding how to properly use Git is very important and beyond the scope of this text. It's recommended that you complete the [2-hour course on Codecademy](https://www.codecademy.com/learn/learn-git) if you've never used Git before or if you've never worked with remote repositories. 

##### 3. Our Branching Model
There are two primary branches in the repository: master and dev. The master branch is ALWAYS stable. Do not push any work directly to the master branch. The dev branch contains work that is being staged to go into the master branch. When the dev branch is determined to be stable and complete, it will be merged into master.

When working on the project, you'll maintain your own branches off of dev (or master), but when it comes time for integration, you must merge into dev.

## Setting up the Project

##### 1. Creating a new project
Now that OpenCV is successfully installed and configured, we can create a new project to work with. Open Visual Studio and choose File > New Project. Choose an "Empty Project" and then choose create.

##### 2. Switching to a 64-bit configuration

By default, the new project will be in a 32 bit configuration, and we built the OpenCV libraries in 64-bit so we'll need to change the configuration. On the menu bar choose Project > Properties. In the top right you'll see "Configuration Manager...". Click it. Look for "Active solution platform" and in the drop down, choose "&lt;New..&gt;".

The "New Solution Platform" window will pop up. Select "x64" as the new platform and then press OK. You can close the Configuration Manager. The platform should now show as x64 in the project property window:

![image](http://i.imgur.com/BDQmyJK.png)

##### 3. Downloading the source

The next step is to download the source code from the repository. We will put the source in a special directory of the project folder so that the project files stay separate from the git repository.

When you create a project (with the name "Keypoints" in this example), it creates the following file structure:

Keypoints/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<-- Solution directory**<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.sdf<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.sln<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.v12.suo<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<-- Project directory**<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.vcxproj<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.vcxproj.filters<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.vcxproj.user<br />

The top level is the solution folder, and the second level is the project folder. We want to add the source code to the project directory. Open Command Prompt and navigate into the project directory (the inner folder that contains the .vcxproj* files). Paste in this command (or substitute your branch for "dev"):
> git clone -b dev https://github.com/cfolson/css487keypoints.git src

It should prompt you for a username and password, and then download all of the contents into a new src directory. Note that you should never commit to the dev (or master) branch. Submit a pull request to the dev branch that will be handled by Prof. Olson. Your file structure should now look something like this:

Keypoints/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<-- Solution directory**<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.sdf<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.sln<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.v12.suo<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<-- Project directory**<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.vcxproj<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.vcxproj.filters<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.vcxproj.user<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;src/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<-- Source directory (also our git repository)**<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<lots of new files in here><br />

##### 4. Setting up the configuration, datasets, and output directories

In the src folder, there should be a folder called "setup" which contains three more folders, "config," "datasets," and "output." Copy these three folders into the Project directory. 

Next, go into the **datasets** directory (the one you just copied over) and unzip the "oxford.zip" file. Your file structure should now look like this:

Keypoints/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<-- Solution directory**<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.sdf<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.sln<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.v12.suo<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<-- Project directory**<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;config/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<-- Configuration directory**<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;config.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<-- Configuration file**<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;datasets/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<-- Dataset directory**<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;dataset directories&gt;<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;illumination-color/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;illumination-direction/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;oxford/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;output/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<-- Output directory**<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.vcxproj<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.vcxproj.filters<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keypoints.vcxproj.user<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;src/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**<-- Source directory (also our git repository)**<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br />

<br />
Information on downloading the **illumination_color** and **illumination_direction** datasets can be found in the [Additional Datasets](#additional-datasets) section.


##### 5. Adding the project files

Now that we've downloaded all of the c++ files, we'll need to add them to our project. Right click on the project in the Solution Explorer and choose Add > Existing Item...

![image](http://i.imgur.com/8B116mu.png)

When the window pops up, navigate to the src/ folder. Then for file type choose "Visual C++ Files (*.c;...". Select all of the files and click "Add".

![image](http://i.imgur.com/PTHElsm.png)

All of the source files have now been added to your project.

##### 6. Setting the property sheet

The last thing we need to do, is tell your project where it can find the OpenCV libraries. Go to the main menu View > Other Windows > Property Manager. Right click on "Debug | x64" and choose "Add exisiting property sheet...". In the src/ file you'll find OPENCV_DEBUG. Select it and click "Open".

##### 7. Build the project!

You should now be able to build and run the project in Debug mode.

<br />

### Using the Configuration File

The program is executed based on the parameters of the configuration file, config.txt (the location of this file is shown in Step 4 of **Setting up the Project**). These parameters are shown here:

dataset: `<dataset>`<br />
imageset: `<imageset1>`, `<imageset2>`, ... `<imagesetN>`<br />
images: `<image1>`, `<image2>`, `<image3>`, ... `<imageN>`<br />
homographies: `<homography 1>`, ... `<homography N-1>`<br />
descriptors: `<descriptor1>`, `<descriptor2>`, ... `<descriptorN>`<br />
extractor: `<extractor>`<br />
save: `<bool>`<br />
display: `<bool>`<br />


#### Specifying Parameters

##### 1. Dataset

The dataset parameter is where you specify which dataset you'll be using. In the current implementation there are three datasets which can be used: illumination-color, illumination-direction, and oxford.


##### 2. Imageset

The imageset parameter is where you specify which imageset(s) you'll be using. You can enter as many imagesets as you like, however, all imagesets which are specified must be located in the specified dataset.


##### 3. Images

The images parameter is where you specify which images you would like to match. You can enter any number of images, given that images of the exact name are located in **all** of the specified imagesets. The first image listed is always matched with all subsequent images. A minimum of two images must be specified.


##### 4. Homography File

The homographies parameter is where you specify which homography files you will be using for matching. The number of homography files specified must be exactly one less than the number of images, as each homography file corresponds to one image matching pair. If three images are specified, then `<homography 1>` corresponds to the `<image1>-<image2>` pair, and `<homography2>` corresponds to the `<image1>-<image3>` pair.


##### 5. Descriptor

The descriptors parameter is where you specify which descriptor(s) to use. You can enter any number of single or stacked descriptors. Stacked descriptors must be entered in the format `<descriptor+descriptor>`, i.e. SIFT+RGBSIFT.


##### 6. Extractor

The extractor parameter is where you specify which feature extractor to use. For the majority of testing, we will be using the SURF feature extractor.


##### 7. Save Data

The save parameter is where you specify if you would like to save the keypoint and descriptor data to a file. To save the data enter "true" for the save parameter (without the quotation marks).


##### 7. Display Matches

The display parameter is where you specify if you would like to display the two images with the matches showing. To display the matches enter "true" for the display parameter (without the quotation marks).


#### Example Configuration File

dataset: oxford<br />
imageset: boat, graf, trees<br />
images: img1.ppm, img2.ppm, img3.ppm, img6.ppm<br />
homographies: H1to2p.txt, H1to3p.txt, H1to6p.txt<br />
descriptors: SIFT, SIFT+RGBSIFT<br />
extractor: SURF<br />
save: false<br />
display: false<br />

This configuration would match img1-img2, img1-img3, and img1-img6 from each the boat, graf, and tree imagesets from the oxford dataset. Matching would be done for all image pairs using both the SIFT and SIFT+RGBSIFT descriptors, and the SURF feature extractor.

<br />

### Running The Test Script

Before any changes can be commited to the GitHub repository, the code must first pass the test script, **testingScript.bat**. This script runs all of the descriptor combinations on a select few datasets and then compares the results to the correct outputs. This ensures that any changes you have made in your local branch have not broken the properly functioning descriptors.

In order to run the test script, simply double-click on **testingScript.bat**, which is located in the **src** directory. **Do not** move the test script, as it assumes this location for proper execution. The script takes a considerable amount of time to run. When the script has completed, it will either return _"The test has PASSED"_ or _"The test has FAILED"_. If the script fails, all issues must be resolved until the script passes before your code can be checked-in to the GitHub repository.



<br />

### Additional Datasets

The **illumination_color** and **illumination_direction** datasets are quite large and cannot be stored in our GitHub reposity. These datasets are available from the [Amsterdam Library of Object Images (ALOI)](http://aloi.science.uva.nl). From the home page, select "Download" from the left menu bar. Once on the download page, select the **Full Color (24-bit), Full Resolution (768 x 576)** _illumination direction (12 GB)_ and _illumination color (6 GB)_ files. These files are large and the downloads can take a considerable amount of time. Once the files are downloaded, move all 1000 imagesets from each into the **illumination_color** and **illumination_direction** dataset directories, respectively.
