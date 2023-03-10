// import '../flutter_flow/flutter_flow_drop_down.dart';
// import '../flutter_flow/flutter_flow_theme.dart';
// import '../flutter_flow/flutter_flow_util.dart';

// ignore_for_file: import_of_legacy_library_into_null_safe, unused_import, prefer_const_constructors

//import 'package:attendancesys_flutter/Screens/LogIn.dart';
import 'dart:convert';

// import 'package:attendance_sys/UI/Pages/AddTrainStudents.dart';
// import 'package:attendance_sys/UI/Pages/AttendanceList.dart';
import 'package:attendance_sys/UI/Pages/AddTrainStudents.dart';
import 'package:attendance_sys/UI/Pages/AttendanceList.dart';
import 'package:attendance_sys/UI/Pages/LogIn.dart';
import 'package:attendance_sys/UI/Pages/StudentInfo.dart';
// import 'package:attendance_sys/UI/Pages/StudentInfo.dart';
// import 'package:attendance_sys/UI/Pages/TakeAttendance.dart';
import 'package:attendance_sys/function.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:hexcolor/hexcolor.dart';
import 'package:provider/provider.dart';
import '../../providers/auth.dart';
import '../../constants.dart';
import './TakeAttendance.dart';
// import './StudentInfo.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({Key? key}) : super(key: key);
  static const routeName = "/dashboard";

  @override
  _DashboardScreenState createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  late String dropDownValue1;
  late String dropDownValue2;
  final scaffoldKey = GlobalKey<ScaffoldState>();
  var _isInit = true;
  var _isLoading = false;
  String? _error;
  String? _fullname;

  Map<String, dynamic> subjects = {};
  String? subChoose;
  String? classsChoose;
  String? attendanceTime;
  // late final url = '';
  List<DropdownMenuItem<String>> getList(lists) {
    List<DropdownMenuItem<String>> dropdownItems = [];
    // print(lists);
    for (String each in lists) {
      var newItem = DropdownMenuItem(
        child: Text(
          each,
          style: TextStyle(
            fontSize: 18,
            fontFamily: 'Roboto',
            color: Color(0xFF265784),
          ),
        ),
        value: each,
      );
      dropdownItems.add(newItem);
    }
    return dropdownItems;
  }

  @override
  void didChangeDependencies() async {
    if (_isInit) {
      setState(() {
        _isLoading = true;
      });
      final token = Provider.of<Auth>(context).token;
      try {
        // fetch classname and subject name
        var data = await fetchData("${BACKEND_URL}/api/userinfo",
            body: {}, method: "GET", token: token);

        setState(() {
          _fullname = data["name"];
          subjects = data["subjects"];
          _isLoading = false;
        });
      } catch (error) {
        setState(() {
          _isLoading = false;
          _error = error as String;
        });
      }

      _isInit = false;
      super.didChangeDependencies();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: scaffoldKey,
      appBar: PreferredSize(
        preferredSize:
            Size.fromHeight(MediaQuery.of(context).size.height * 0.14),
        child: SingleChildScrollView(
          child: AppBar(
            backgroundColor: const Color(0xFF265784),
            automaticallyImplyLeading: false,
            flexibleSpace: Container(
              padding: EdgeInsets.only(right: 20, bottom: 10),
              child: Column(
                mainAxisSize: MainAxisSize.max,
                children: [
                  const Align(
                    alignment: AlignmentDirectional(-0.5, 0),
                    child: Padding(
                      padding: EdgeInsetsDirectional.fromSTEB(0, 45, 0, 0),
                      child: Text(
                        'Smart Attendance NCE',
                        style: TextStyle(
                          fontFamily: 'Poppins',
                          color: Colors.white,
                          fontSize: 30,
                        ),
                      ),
                    ),
                  ),
                  Align(
                    alignment: const AlignmentDirectional(0.95, 0),
                    child: Container(
                      width: MediaQuery.of(context).size.width * 0.2,
                      height: MediaQuery.of(context).size.height * 0.015,
                      decoration: const BoxDecoration(
                        color: Color(0xFF265784),
                      ),
                    ),
                  ),
                  Align(
                    alignment: Alignment.centerRight,
                    // const AlignmentDirectional(0.85, 0),
                    child: InkWell(
                      onTap: () async {
                        await showDialog(
                          context: context,
                          builder: (alertDialogContext) {
                            return AlertDialog(
                              title: Text('Are you sure to logout?'),
                              actions: [
                                TextButton(
                                  onPressed: () =>
                                      Navigator.pop(alertDialogContext),
                                  child: Text('No'),
                                ),
                                TextButton(
                                  onPressed: () async {
                                    Navigator.pop(alertDialogContext);
                                    await Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                        builder: (context) => const LogInScreen(),
                                      ),
                                    );
                                  },
                                  child: Text('Yes'),
                                ),
                              ],
                            );
                          },
                        );
                      },
                      child: Icon(
                        Icons.logout,
                        color: Colors.white,
                        size: 25,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            actions: [
              Align(
                alignment: AlignmentDirectional(0, 0.15),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(0),
                  child: Image.asset(
                    'assets/images/2123.png',
                    width: MediaQuery.of(context).size.width * 0.2,
                    fit: BoxFit.contain,
                  ),
                ),
              ),
            ],
            elevation: 0,
          ),
        ),
      ),
      backgroundColor: Colors.white,
      body: _isLoading
          ? CircularProgressIndicator()
          : SafeArea(
              child: GestureDetector(
                onTap: () => FocusScope.of(context).unfocus(),
                child: Padding(
                  padding: EdgeInsetsDirectional.fromSTEB(1, 0, 0, 0),
                  child: Column(
                    mainAxisSize: MainAxisSize.max,
                    mainAxisAlignment: MainAxisAlignment.start,
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      SingleChildScrollView(
                        child: Column(
                          mainAxisSize: MainAxisSize.max,
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            Container(
                              width: MediaQuery.of(context).size.width,
                              height: MediaQuery.of(context).size.height * 0.05,
                              decoration: BoxDecoration(
                                color: const Color(0xFF265784),
                                borderRadius: BorderRadius.circular(0),
                                border: Border.all(
                                  color: Colors.white,
                                  width: 0,
                                ),
                              ),
                              child: Container(
                                width: MediaQuery.of(context).size.width,
                                height: MediaQuery.of(context).size.height * 1,
                                decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: const BorderRadius.only(
                                    bottomLeft: Radius.circular(0),
                                    bottomRight: Radius.circular(0),
                                    topLeft: Radius.circular(100),
                                    topRight: Radius.circular(0),
                                  ),
                                  border: Border.all(
                                    color: Colors.white,
                                  ),
                                ),
                              ),
                            ),
                            Column(
                              mainAxisSize: MainAxisSize.max,
                              children: [
                                Align(
                                  alignment: AlignmentDirectional(-0.6, 0),
                                  child: Text(
                                    "$_fullname",
                                    style: TextStyle(
                                      fontFamily: 'Roboto',
                                      color: Color(0xFF265784),
                                      fontSize: 18,
                                      fontWeight: FontWeight.normal,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                      Padding(
                        padding:
                            const EdgeInsetsDirectional.fromSTEB(0, 10, 0, 0),
                        child: SizedBox(
                          width: 230,
                          child: DropdownButton<String>(
                            hint: Text(
                              'Choose Subject',
                              style: TextStyle(
                                color: HexColor('#265784'),
                              ),
                            ),
                            isExpanded: true,
                            value: subChoose,
                            //icon: const Icon(Icons.arrow_downward),
                            elevation: 16,
                            style: TextStyle(color: HexColor('#265784')),
                            underline: Container(
                              height: 2,
                              color: HexColor('#265784'),
                            ),
                            onChanged: (String? newValue) {
                              setState(() {
                                subChoose = newValue!;
                                classsChoose = subChoose != null &&
                                        classsChoose != null &&
                                        subjects[subChoose]
                                            .contains(classsChoose)
                                    ? classsChoose
                                    : null;
                              });
                            },
                            items: getList(subjects.keys.toList()),
                          ),
                        ),
                      ),
                      Padding(
                          padding:
                              const EdgeInsetsDirectional.fromSTEB(0, 10, 0, 0),
                          child: SizedBox(
                            width: 230,
                            child: DropdownButton<String>(
                                hint: Text(
                                  'Choose class',
                                  style: TextStyle(
                                    color: HexColor('#265784'),
                                  ),
                                ),
                                isExpanded: true,
                                value: classsChoose,
                                //icon: const Icon(Icons.arrow_downward),
                                elevation: 16,
                                style: TextStyle(color: HexColor('#265784')),
                                underline: Container(
                                  height: 2,
                                  color: HexColor('#265784'),
                                ),
                                onChanged: (String? newValue) {
                                  setState(() {
                                    classsChoose = newValue!;
                                  });
                                },
                                items: getList(subChoose != null
                                    ? subjects[subChoose]
                                    : [])),
                          )),
                      Expanded(
                        child: Padding(
                          padding:
                              const EdgeInsetsDirectional.fromSTEB(0, 30, 0, 0),
                          child: GridView(
                            padding: EdgeInsets.zero,
                            gridDelegate:
                                const SliverGridDelegateWithFixedCrossAxisCount(
                              crossAxisCount: 2,
                              crossAxisSpacing: 20,
                              mainAxisSpacing: 20,
                              childAspectRatio: 1,
                            ),
                            scrollDirection: Axis.vertical,
                            children: [
                              Padding(
                                padding: const EdgeInsetsDirectional.fromSTEB(
                                    20, 20, 0, 0),
                                child: Card(
                                  clipBehavior: Clip.antiAliasWithSaveLayer,
                                  color: Colors.white,
                                  elevation: 10,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(20),
                                  ),
                                  child: InkWell(
                                    onTap: () {
                                      showDialog(
                                          context: context,
                                          builder: (BuildContext context) {
                                            return AlertDialog(
                                              scrollable: true,
                                              content: Padding(
                                                padding:
                                                    const EdgeInsets.all(4.0),
                                                child: Form(
                                                  child: Column(
                                                    children: <Widget>[
                                                      TextFormField(
                                                        decoration:
                                                            InputDecoration(
                                                          labelText:
                                                              'Attendance Time',
                                                          icon:
                                                              Icon(Icons.watch),
                                                        ),
                                                        onChanged:
                                                            (String? newValue) {
                                                          setState(() {
                                                            attendanceTime =
                                                                newValue!;
                                                          });
                                                        },
                                                      ),
                                                      TextFormField(
                                                        decoration:
                                                            InputDecoration(
                                                          labelText:
                                                              'Camera Source',
                                                          icon: Icon(
                                                              Icons.source),
                                                        ),
                                                      ),
                                                    ],
                                                  ),
                                                ),
                                              ),
                                              actions: [
                                                RaisedButton(
                                                  child: Text("Start"),
                                                  onPressed: () async {
                                                    var body = {
                                                      "classname": classsChoose,
                                                      "subjectname": subChoose,
                                                      "time": attendanceTime,
                                                    };
                                                    // Navigator.of(context).pop();
                                                    Navigator.of(context)
                                                        .pushReplacementNamed(
                                                            TakeAttendanceScreen
                                                                .routeName,
                                                            arguments: body);
                                                  },
                                                )
                                              ],
                                            );
                                          });
                                    },
                                    child: Column(
                                      mainAxisSize: MainAxisSize.max,
                                      mainAxisAlignment:
                                          MainAxisAlignment.spaceEvenly,
                                      children: const [
                                        Align(
                                          alignment: AlignmentDirectional(0, 0),
                                          child: FaIcon(
                                            FontAwesomeIcons.calendarAlt,
                                            color: Color(0xFFEA734D),
                                            size: 60,
                                          ),
                                        ),
                                        Text(
                                          'Take\nAttendance',
                                          textAlign: TextAlign.center,
                                          style: TextStyle(
                                            fontFamily: 'Poppins',
                                            color: Color(0xFFEA734D),
                                            fontSize: 18,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                              Padding(
                                padding: const EdgeInsetsDirectional.fromSTEB(
                                    0, 20, 20, 0),
                                child: Card(
                                  clipBehavior: Clip.antiAliasWithSaveLayer,
                                  color: Colors.white,
                                  elevation: 10,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(20),
                                  ),
                                  child: InkWell(
                                    onTap: () {
                                      var body = {
                                        "classname": classsChoose,
                                        "subjectname": subChoose,
                                      };
                                      Navigator.of(context).pushNamed(
                                          AttendanceListScreen.routeName,
                                          arguments: body);
                                    },
                                    child: Column(
                                      mainAxisSize: MainAxisSize.max,
                                      mainAxisAlignment:
                                          MainAxisAlignment.spaceEvenly,
                                      children: const [
                                        Align(
                                          alignment:
                                              AlignmentDirectional(-0.05, 0),
                                          child: FaIcon(
                                            FontAwesomeIcons.eye,
                                            color: Color(0xFFEA734D),
                                            size: 60,
                                          ),
                                        ),
                                        Text(
                                          'View\nAttendance',
                                          textAlign: TextAlign.center,
                                          style: TextStyle(
                                            fontFamily: 'Poppins',
                                            color: Color(0xFFEA734D),
                                            fontSize: 18,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                              Padding(
                                padding: const EdgeInsetsDirectional.fromSTEB(
                                    20, 0, 0, 20),
                                child: InkWell(
                                  onTap: () {
                                    Navigator.of(context).pushNamed(
                                      AddTrainStudentScreen.routeName,
                                    );
                                  },
                                  child: Card(
                                    clipBehavior: Clip.antiAliasWithSaveLayer,
                                    color: Colors.white,
                                    elevation: 10,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(20),
                                    ),
                                    child: Column(
                                      mainAxisSize: MainAxisSize.max,
                                      mainAxisAlignment:
                                          MainAxisAlignment.spaceEvenly,
                                      children: const [
                                        Icon(
                                          Icons.add_circle,
                                          color: Color(0xFFEA734D),
                                          size: 60,
                                        ),
                                        Text(
                                          'Add/Train\nStudents',
                                          textAlign: TextAlign.center,
                                          style: TextStyle(
                                            fontFamily: 'Poppins',
                                            color: Color(0xFFEA734D),
                                            fontSize: 18,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                              Padding(
                                padding: const EdgeInsetsDirectional.fromSTEB(
                                    0, 0, 20, 20),
                                child: Card(
                                  clipBehavior: Clip.antiAliasWithSaveLayer,
                                  color: Colors.white,
                                  elevation: 10,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(20),
                                  ),
                                  child: InkWell(
                                    onTap: () async {
                                      var body = {
                                        "classname": classsChoose,
                                        "subjectname": subChoose,
                                      };
                                      Navigator.of(context).pushNamed(
                                          StudentInfoScreen.routeName,
                                          arguments: body);
                                    },
                                    child: Column(
                                      mainAxisSize: MainAxisSize.max,
                                      mainAxisAlignment:
                                          MainAxisAlignment.spaceEvenly,
                                      children: const [
                                        Icon(
                                          Icons.contact_page_sharp,
                                          color: Color(0xFFEA734D),
                                          size: 60,
                                        ),
                                        Text(
                                          'Student\nInfo',
                                          textAlign: TextAlign.center,
                                          style: TextStyle(
                                            fontFamily: 'Poppins',
                                            color: Color(0xFFEA734D),
                                            fontSize: 18,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
    );
  }
}
