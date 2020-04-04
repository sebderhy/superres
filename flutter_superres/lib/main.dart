import 'package:flutter/material.dart';
import 'package:fluttersuperres/image_compare.dart';
import 'home_screen.dart';

void main() => runApp(Superres());

class Superres extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      initialRoute: HomeScreen.id,
      routes: {
        HomeScreen.id: (context) => HomeScreen(),
//        ImageCompare.id: (context) => ImageCompare(
//              text1: "from ",
//              text2: "main screen !",
//            ),
      },
    );
  }
}

//
//void main() {
//  runApp(new MaterialApp(
//    debugShowCheckedModeBanner: false,
//    title: "Superres",
//    home: new HomeScreen(),
//  ));
//}
