/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Viewer.h"
#include <pangolin/pangolin.h>

#include <mutex>

#include "ransac_estimator.cpp"

namespace ORB_SLAM2 {

    Viewer::Viewer(System *pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking,
                   const string &strSettingPath) :
      mpSystem(pSystem), mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpTracker(pTracking),
      mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false) {
      cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

      float fps = fSettings["Camera.fps"];
      if (fps < 1)
        fps = 30;
      mT = 1e3 / fps;

      mImageWidth = fSettings["Camera.width"];
      mImageHeight = fSettings["Camera.height"];
      if (mImageWidth < 1 || mImageHeight < 1) {
        mImageWidth = 640;
        mImageHeight = 480;
      }

      mViewpointX = fSettings["Viewer.ViewpointX"];
      mViewpointY = fSettings["Viewer.ViewpointY"];
      mViewpointZ = fSettings["Viewer.ViewpointZ"];
      mViewpointF = fSettings["Viewer.ViewpointF"];
    }

    void Viewer::Run() {
      mbFinished = false;
      mbStopped = false;

      pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer", 1024, 768);

      // 3D Mouse handler requires depth testing to be enabled
      glEnable(GL_DEPTH_TEST);

      // Issue specific OpenGl we might need
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
      pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
      pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
      pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
      pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
      pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode", false, true);
      pangolin::Var<bool> menuReset("menu.Reset", false, false);

      // Define Camera Render Object (for view / scene browsing)
      pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
      );

      // Add named OpenGL viewport to window and provide 3D Handler
      pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

      pangolin::OpenGlMatrix Twc;
      Twc.SetIdentity();

      cv::namedWindow("ORB-SLAM2: Current Frame");
      cv::namedWindow("ORB-SLAM2: OF Frame");

      bool bFollow = true;
      bool bLocalizationMode = false;

      int counter1 = 0;
      //int hog[8];
      std::array<int, 8> hog;
      hog.fill(0);
      int hog_index[8] = {0, 45, 90, 135, 180, 225, 270, 315};
      double hog_scale = 0.01;
      double ransac_scale = 50;
      int phi = 0;
      int i = 0;

      cv::Mat ims;
      cv::Mat orginal;

      //cv::Size frameSize(static_cast<int>(mImageWidth), static_cast<int>(mImageHeight));
      //cv::Size frameSizeMask(static_cast<int>(mImageWidth/10), static_cast<int>(mImageHeight/10));
      //cv::VideoWriter videoOF("out_OF.avi",CV_FOURCC('P','I','M','1'),30,frameSize, true);
      //cv::VideoWriter videoRansac1("out_RANSAC1.avi",CV_FOURCC('P','I','M','1'),20,frameSizeMask, false);
      //cv::VideoWriter videoRansac2("out_RANSAC2.avi",CV_FOURCC('P','I','M','1'),20,frameSizeMask, false);

      while (1) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

        if (menuFollowCamera && bFollow) {
          s_cam.Follow(Twc);
        } else if (menuFollowCamera && !bFollow) {
          s_cam.SetModelViewMatrix(
            pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
          s_cam.Follow(Twc);
          bFollow = true;
        } else if (!menuFollowCamera && bFollow) {
          bFollow = false;
        }

        if (menuLocalizationMode && !bLocalizationMode) {
          mpSystem->ActivateLocalizationMode();
          bLocalizationMode = true;
        } else if (!menuLocalizationMode && bLocalizationMode) {
          mpSystem->DeactivateLocalizationMode();
          bLocalizationMode = false;
        }

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        mpMapDrawer->DrawCurrentCamera(Twc);
        if (menuShowKeyFrames || menuShowGraph)
          mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
        if (menuShowPoints)
          mpMapDrawer->DrawMapPoints();

        pangolin::FinishFrame();

        cv::Mat im = mpFrameDrawer->DrawFrame();
        cv::imshow("ORB-SLAM2: Current Frame", im);


        cv::Mat im2 = mpFrameDrawer->DrawFrameOptFlow();
        //cv::imshow("ORB-SLAM2: OF Frame",im2);

        cv::Mat grayIm;
        cv::resize(im2, grayIm, cv::Size(640, 480));
        cv::cvtColor(im2, grayIm, cv::COLOR_BGR2GRAY);
        if (counter1 == 0) {

          ims = grayIm;

        }


        orginal = im2;

        cv::Mat flow;
        cv::UMat flowUmat;
        cv::calcOpticalFlowFarneback(ims, grayIm, flowUmat, 0.5, 2, 24, 2, 7, 1.5, 0);
        flowUmat.copyTo(flow);


        hog.fill(0);
        // Tegn gradientvektorer på orginal bildet
        for (int y = 0; y < orginal.rows; y += 20) {
          for (int x = 0; x < orginal.cols; x += 20) {
            // * 2 for synlighet
            const cv::Point2f flowatxy = flow.at<cv::Point2f>(y, x) * 2;
            // Tegn vektor for gradient
            cv::arrowedLine(orginal, cv::Point(x, y), cv::Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)),
                            cv::Scalar(255, 0, 0),2);

            // Summer HOG for viste gradienter
            if (flowatxy.y != 0) { phi = atan(flowatxy.x / flowatxy.y) * 180 / 3.14159; }
            else {
              phi = atan((flowatxy.x + 1) / (flowatxy.y + 1)) * 180 / 3.14159;
            }
            if (phi < 0) { phi *= -1; }
            if ((flowatxy.x < 0) & (flowatxy.y >= 0)) { phi += 90; }
            else if (flowatxy.x < 0) { phi += 180; }
            else if (flowatxy.y < 0) { phi += 270; }
            if (phi > 315) { i = 7; }
            else if (phi > 270) { i = 6; }
            else if (phi > 225) { i = 5; }
            else if (phi > 180) { i = 4; }
            else if (phi > 135) { i = 3; }
            else if (phi > 90) { i = 2; }
            else if (phi > 45) { i = 1; } else { i = 0; }
            hog[i] += sqrt(flowatxy.x * flowatxy.x + flowatxy.y * flowatxy.y);
          }
        }

        // Skriver ut HOG for alle HOG_index-verdier
        /*for (int j = 0; j < 8; j += 1) {
          cv::arrowedLine(orginal, cv::Point(orginal.cols / 2, orginal.rows / 2),
                   cv::Point((orginal.cols / 2) + cos(hog_index[j] * 3.14159 / 180 + 22.5) * hog[j] * hog_scale,
                             (orginal.rows / 2) + sin(hog_index[j] * 3.14159 / 180 + 22.5) * hog[j] * hog_scale),
                   cv::Scalar(0, 255, 0), 5,8,0,0.3);
        }*/

        // Kjøre RANSAC for å finne scenen
        RansacEstimator estimator1(0.99,2.5f,1,3.0f,10);
        RansacEstimate estimate1 = estimator1.estimate(flow);


        // Skriver ut vektor fra første RANSAC
        cv::arrowedLine(orginal, cv::Point(orginal.cols / 2, orginal.rows / 2),
                   cv::Point((orginal.cols / 2) + estimate1.best_vector(0) * ransac_scale, (orginal.rows / 2) + estimate1.best_vector(1) * ransac_scale),
                   cv::Scalar(0, 0, 255), 5,8,0,0.3);


        // Kjøre RANSAC for å finne objektet
        RansacEstimator estimator2(0.99,0.9f,0.7,2.0f,10);
        RansacEstimate estimate2 = estimator2.estimate_with_mask(flow,estimate1.is_inler_mat);


        /*for (int y = 0; y <= (estimate1.is_inler_mat.rows-1); y += 1) {
          for (int x = 0; x <= (estimate1.is_inler_mat.cols - 1); x += 1) {
            if (estimate1.is_inler_mat.at<char>(y,x) == 0) {
              cv::line(orginal, cv::Point(x*10, y*10),
                       cv::Point(x*10 + 1, y*10 + 1),
                       cv::Scalar(0, 0, 255), 2);
            }
          }
        }*/




        // Skriver ut inneliggere fra første RANSAC
        cv::namedWindow("ORB-SLAM2: RANSAC 1 Frame",CV_WINDOW_FREERATIO);
        cv::imshow("ORB-SLAM2: RANSAC 1 Frame", estimate1.is_inler_mat);

        // Skriver ut inneliggere fra andre RANSAC
        cv::namedWindow("ORB-SLAM2: RANSAC 2 Frame",CV_WINDOW_FREERATIO);
        cv::imshow("ORB-SLAM2: RANSAC 2 Frame", estimate2.is_inler_mat);

        cv::imshow("ORB-SLAM2: OF Frame", orginal);

        //videoOF.write(orginal);
        //videoRansac1.write(estimate1.is_inler_mat);
        //videoRansac2.write(estimate2.is_inler_mat);

        ims = grayIm;

        counter1++;


        cv::waitKey(mT);

        if (menuReset) {
          menuShowGraph = true;
          menuShowKeyFrames = true;
          menuShowPoints = true;
          menuLocalizationMode = false;
          if (bLocalizationMode)
            mpSystem->DeactivateLocalizationMode();
          bLocalizationMode = false;
          bFollow = true;
          menuFollowCamera = true;
          mpSystem->Reset();
          menuReset = false;
        }

        if (Stop()) {
          while (isStopped()) {
            usleep(3000);
          }
        }

        if (CheckFinish())
          break;
      }

      SetFinish();
    }

    void Viewer::RequestFinish() {
      unique_lock<mutex> lock(mMutexFinish);
      mbFinishRequested = true;
    }

    bool Viewer::CheckFinish() {
      unique_lock<mutex> lock(mMutexFinish);
      return mbFinishRequested;
    }

    void Viewer::SetFinish() {
      unique_lock<mutex> lock(mMutexFinish);
      mbFinished = true;
    }

    bool Viewer::isFinished() {
      unique_lock<mutex> lock(mMutexFinish);
      return mbFinished;
    }

    void Viewer::RequestStop() {
      unique_lock<mutex> lock(mMutexStop);
      if (!mbStopped)
        mbStopRequested = true;
    }

    bool Viewer::isStopped() {
      unique_lock<mutex> lock(mMutexStop);
      return mbStopped;
    }

    bool Viewer::Stop() {
      unique_lock<mutex> lock(mMutexStop);
      unique_lock<mutex> lock2(mMutexFinish);

      if (mbFinishRequested)
        return false;
      else if (mbStopRequested) {
        mbStopped = true;
        mbStopRequested = false;
        return true;
      }

      return false;

    }

    void Viewer::Release() {
      unique_lock<mutex> lock(mMutexStop);
      mbStopped = false;
    }

}

