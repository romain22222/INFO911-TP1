#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>

using namespace cv;

std::vector<double> histogramme( Mat image){
  std::vector<double> hist(256,0);
  for(int i = 0; i < image.rows; i++){
    for(int j = 0; j < image.cols; j++){
      hist[image.at<uchar>(i,j)] += 1.0 / (image.rows*image.cols) ;
    }
  }
  return hist;
}

std::vector<double> histogramme_cumule(const std::vector<double>& h_I){
  std::vector<double> h_C(256,0);
  h_C[0] = h_I[0];
  for(int i = 1; i < 256; i++){
    h_C[i] = h_C[i-1] + h_I[i];
  }
  return h_C;
}

Mat afficheHistogrammes( const std::vector<double>& h_I, const std::vector<double>& H_I){
  Mat histImage( 256 , 512, CV_8U, Scalar(0) );
  normalize(h_I, h_I, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(H_I, H_I, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  //get max value of h_i
  double max = 0;
  for(int i = 0; i < 256; i++){
    if(h_I[i] > max){
      max = h_I[i];
    }
  }

  //histogramme
  for(int i = 1; i < max; i++){
    for (int j = 1; j < 256; j++){
      if(h_I[j] < i){
        histImage.at<uchar>(Point(j, 256-i)) = 255;
      }
    }
  }

  //histogramme cumulé
  for(int i = 1; i < max; i++){
    for (int j = 1; j < 256; j++){
      if(H_I[j] < i){
        histImage.at<uchar>(Point(j+255, 256-i)) = 255;
      }
    }
  }
  imshow("histogramme", histImage);
  return histImage;
}

std::vector<std::vector<double>> createHistogram(Mat image){
  std::vector<double> h_I = histogramme(image);
  std::vector<double> H_I = histogramme_cumule(h_I);
  //retourne h_I et H_I
  std::vector<std::vector<double>> result;
  result.push_back(h_I);
  result.push_back(H_I);
  return result;
  // Mat histogramme = afficheHistogrammes(h_I, H_I);
  // imshow(name, histogramme);
}

Mat egalisation(const Mat& image){
  std::vector<std::vector<double>> h = createHistogram(image);

  Mat new_image = image.clone();
  for(int i = 0; i < image.rows; i++){
    for(int j = 0; j < image.cols; j++){
      new_image.at<uchar>(i,j) = 255 * h[1][image.at<uchar>(i,j)];
    }
  }
  return new_image;
}

Mat egalisationAvecCouleur(const Mat& image){
  Mat hsv_image;
  cvtColor(image, hsv_image, COLOR_BGR2HSV);
  std::vector<Mat> hsv_channels;
  split(hsv_image, hsv_channels);
  hsv_channels[2] = egalisation(hsv_channels[2]);
  Mat new_image;
  merge(hsv_channels, new_image);
  cvtColor(new_image, new_image, COLOR_HSV2BGR);
  return new_image;
}

void exercise1(Mat image){
  cvtColor(image, image, COLOR_BGR2GRAY);
  imshow( "TP1", image );                // l'affiche dans la fenêtre

  std::vector h = createHistogram(image);
  Mat histogramme = afficheHistogrammes(h[0], h[1]);
}

void exercise2(Mat image){
  cvtColor(image, image, COLOR_BGR2GRAY);
  Mat new_image = egalisation(image);
  imshow("TP1", new_image);

  std::vector h = createHistogram(new_image);
  Mat histogramme = afficheHistogrammes(h[0], h[1]);
}

void exercise3(Mat image){
  Mat new_image = egalisationAvecCouleur(image);
  imshow("TP1", new_image);

  //transform to hsv
  Mat hsv_image;
  cvtColor(new_image, hsv_image, COLOR_BGR2HSV);
  std::vector<Mat> hsv_channels;
  split(hsv_image, hsv_channels);
  std::vector h = createHistogram(hsv_channels[2]);
  Mat histogramme = afficheHistogrammes(h[0], h[1]);
}

void editBW(Mat image, int y, int x, float error){
  auto before = image.at<uchar>(y,x);
  image.at<uchar>(y,x) += error ;
  if(image.at<uchar>(y,x) < before && error > 0){
    image.at<uchar>(y,x) = 255;
  } else if(image.at<uchar>(y,x) > before && error < 0){
    image.at<uchar>(y,x) = 0;
  }
}


void editColor(Mat image, int y, int x, Vec3f error){
  auto before = image.at<Vec3f>(y,x);
  image.at<Vec3f>(y,x) += error ;
  for (int i = 0; i < 3; i++){
    if(image.at<Vec3f>(y,x) [i] < before[i] && error[i] > 0){
      image.at<Vec3f>(y,x)[i] = 1;
    } else if(image.at<Vec3f>(y,x) [i] > before[i] && error[i] < 0){
      image.at<Vec3f>(y,x)[i] = 0;
    }
  }
}
Mat tramage_floyd_steinberg( Mat input){
  std::vector<Mat> input_channels;
  split(input, input_channels);
  Mat output;
  if (input.type() == CV_8U){
    output = Mat(input.rows, input.cols, CV_8U);
  } else {
    output = Mat(input.rows, input.cols, CV_8UC3);
  }
  std::vector<Mat> output_channels;
  split(output, output_channels);
  for(int i = 0; i < input.rows; i++){
    for(int j = 0; j < input.cols; j++){
      for(int c =0; c < input_channels.size(); c++){
        int old_pixel = input_channels[c].at<uchar>(i,j);
        int new_pixel = old_pixel > 128 ? 255 : 0;
        output_channels[c].at<uchar>(i,j) = new_pixel;
        int error = old_pixel - new_pixel;
        if(j < input.cols - 1){
          editBW(input_channels[c], i,j+1, error * 7 / 16);
        }
        if(i < input.rows - 1){
          if(j > 0){
            editBW(input_channels[c], i+1,j-1, error * 3 / 16);
          }
          editBW(input_channels[c], i+1,j, error * 5 / 16);
          if(j < input.cols - 1){
            editBW(input_channels[c], i+1,j+1, error * 1 / 16);
          }
        }
      }
    }
  }
  merge(output_channels, output);
  return output;
}

float distance_color_l2( Vec3f bgr1, Vec3f bgr2 ){
  float val = 0;
  for (int i = 0; i < 3 ; i++){
    val += std::pow(bgr1[i] - bgr2[i], 2);
  }
  return std::sqrt(val);
}

int best_color( Vec3f bgr, std::vector< Vec3f > colors ){
  float best = distance_color_l2(colors[0], bgr);
  int best_index = 0;
  for (int i = 1; i < colors.size(); i++){
    float tmp = distance_color_l2(bgr, colors[i]);
    if (tmp < best){
      best = tmp;
      best_index = i;
    }
  }
  return best_index;
}
Vec3f error_color( Vec3f bgr1, Vec3f bgr2 ){
  return bgr1 - bgr2;
}

Mat tramage_floyd_steinberg( Mat input, std::vector< Vec3f > colors ){
  // Conversion de input en une matrice de 3 canaux flottants
  Mat fs;
  input.convertTo( fs, CV_32FC3, 1/255.0);
  // Algorithme Floyd-Steinberg
  // Pour chaque pixel (x,y) Faire
  for ( int y = 0; y < fs.rows; y++ ){
    for( int x = 0; x < fs.cols; x++){
      Vec3f c = fs.at< Vec3f >(y,x);
      int i = best_color( c, colors );
      Vec3f e = error_color( c, colors[ i ] );
      fs.at< Vec3f >(y,x) = colors[ i ];
      if(x < input.cols - 1){
        editColor(fs, y, x+1, e * 7 / 16);
      }
      if(y < input.rows - 1){
        if(x > 0){
          editColor(fs, y+1, x-1, e * 3 / 16);
        }
        editColor(fs, y+1, x, e * 5 / 16);
        if(x < input.cols - 1){
          editColor(fs, y+1,x+1, e * 1 / 16);
        }
      }
    }
  }
  
  // On reconvertit la matrice de 3 canaux flottants en BGR
  Mat output;
  fs.convertTo( output, CV_8UC3, 255.0 );
  return output;
}

void exercise4(Mat image){
  cvtColor(image, image, COLOR_BGR2GRAY);
  auto new_image = tramage_floyd_steinberg(image);
  imshow("TP1", new_image);
}

void exercise5(Mat image){
  Mat new_image = tramage_floyd_steinberg(image);
  imshow("TP1", new_image);
}

void exercise6(Mat image){
  Mat new_image = tramage_floyd_steinberg(image, {Vec3f(0,0,0), Vec3f(0,1,1), Vec3f(1,0,1), Vec3f(1,1,0), Vec3f(1,1,1)});
  imshow("TP1", new_image);
}


void filter(int exercise, Mat image){
  switch(exercise){
    case 1:
      exercise1(image);
      break;
    case 2:
      exercise2(image);
      break;
    case 3:
      exercise3(image);
      break;
    case 4:
      exercise4(image);
      break;
    case 5:
      exercise5(image);
      break;
    case 6:
      exercise6(image);
      break;
    default:
      std::cout << "Exercise " << exercise << " not found" << std::endl;
  }
}

int main(int argc, char *argv[])
{
  if(argc < 3)
  {
    std::cout << "Usage: " << argv[0] << " <image> <exercise>" << std::endl;
    return 1;
  }

  int value = 128;
  int old_value = value;
  namedWindow("TP1");               // crée une fenêtre
  createTrackbar( "track", "TP1", &value , 255, NULL); // un slider

  int exercise = atoi(argv[2]);
  std::string argv1 = argv[1];
  if (argv1.compare("video") == 0){
    VideoCapture cap(0);
    if(!cap.isOpened()) return -1;
    Mat frame;
    for(;;)
    {
        cap >> frame;
        filter(exercise, frame);
        int   key_code = waitKey(30);
        int ascii_code = key_code & 0xff; 
        if( ascii_code == 'q') break;
    }
  } else{
    Mat image = imread(argv[1]);        // lit l'image "lena.png"
    filter(exercise, image);
  }
  
  while ( waitKey(50) < 0 )          // attend une touche
  { // Affiche la valeur du slider
    if ( value != old_value )
    {
      std::cout << "value=" << value << std::endl;
      old_value = value;
    }
  }
}