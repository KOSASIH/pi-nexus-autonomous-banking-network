import 'package:flutter/material.dart';

class NexusMobileApp extends StatefulWidget {
  @override
  _NexusMobileAppState createState() => _NexusMobileAppState();
}

class _NexusMobileAppState extends State<NexusMobileApp> {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Nexus Mobile App',
      home: Scaffold(
        appBar: AppBar(
          title: Text('Nexus Mobile App'),
        ),
        body: Center(
          child: ElevatedButton(
            child: Text('Login'),
            onPressed: () {
              // Implement login logic
            },
          ),
        ),
      ),
    );
  }
}
