import 'dart:convert';
import 'dart:ui';
import 'package:image_picker/image_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'package:path/path.dart';
import 'package:async/async.dart';
import 'dart:typed_data';

const STATUS_WAIT = 0;
const STATUS_IMAGE_LOADED = 1;
const STATUS_FINISHED = 2;

void main() {
  runApp(new MaterialApp(
    debugShowCheckedModeBanner: false,
    title: "Superres",
    home: new MyApp(),
  ));
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  File img;
  Uint8List derotatedImage;
  int status = STATUS_WAIT;
  String strResponse = '';

  // The function which will upload the image as a file
  void upload(File imageFile) async {
    var stream =
        new http.ByteStream(DelegatingStream.typed(imageFile.openRead()));
    var length = await imageFile.length();

    String base = "http://fastaiserve.com/";

    var uri = Uri.parse(base + 'superres-1b/img2img/');

    var request = new http.MultipartRequest("POST", uri);
    var multipartFile = new http.MultipartFile('file', stream, length,
        filename: basename(imageFile.path));

    request.files.add(multipartFile);
    var response = await request.send();
    derotatedImage = await response.stream.toBytes();

    setState(() {
      status = STATUS_FINISHED;
    });
  }

  void image_picker(int a) async {
    setState(() {});
    debugPrint("Image Picker Activated");
    if (a == 0) {
      img = await ImagePicker.pickImage(source: ImageSource.camera);
    } else {
      img = await ImagePicker.pickImage(source: ImageSource.gallery);
    }

//    txt = "Analyzing...";
    debugPrint(img.toString());
    upload(img);
    setState(() {
      status = STATUS_IMAGE_LOADED;
    });
  }

  Widget textComments(BuildContext context) {
    String comment = '';
    switch (status) {
      case STATUS_WAIT:
        comment = "";
        break;
      case STATUS_IMAGE_LOADED:
        comment = "Enhancing image";
        break;
      case STATUS_FINISHED:
        comment = "Here is the enhanced image!";
        break;
    }

    return Center(
      child: new Text(
        comment,
        textAlign: TextAlign.center,
        style: TextStyle(
          fontWeight: FontWeight.bold,
          fontSize: 24.0,
        ),
      ),
    );
  }

  Widget result(BuildContext context) {
    if (derotatedImage != null) {
      return Image.memory(derotatedImage);
    } else {
      if (img != null) {
        return Image.file(img);
      } else {
        return Center(
          child: new Text(
            "Upload/capture the rotated image you want to enhance",
            textAlign: TextAlign.center,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 24.0,
            ),
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: new AppBar(
        centerTitle: true,
        title: new Text("Superres"),
      ),
      body: new Container(
        padding: EdgeInsets.symmetric(vertical: 50, horizontal: 25),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: <Widget>[
              result(context),
              textComments(context),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: <Widget>[
                  new FloatingActionButton(
                    onPressed: () {
                      image_picker(0);
                    },
                    child: new Icon(Icons.camera_alt),
                  ),
                  new FloatingActionButton(
                      onPressed: () {
                        image_picker(1);
                      },
                      child: new Icon(Icons.file_upload)),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
