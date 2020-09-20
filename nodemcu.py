#Code for Sending Data to Internet using NodeMCU


ESP8266 --> ThingSpeak Channel
 
 This sketch sends the Wi-Fi Signal Strength (RSSI) of an ESP8266 to a ThingSpeak
 channel using the ThingSpeak API (https://www.mathworks.com/help/thingspeak).
 
 Requirements:
 
   * ESP8266 Wi-Fi Device
   * Arduino 1.8.8+ IDE
   * Additional Boards URL: http://arduino.esp8266.com/stable/package_esp8266com_index.json
   * Library: esp8266 by ESP8266 Community
   * Library: ThingSpeak by MathWorks
 
 ThingSpeak Setup:
 
   * Sign Up for New User Account - https://thingspeak.com/users/sign_up
   * Create a new Channel by selecting Channels, My Channels, and then New Channel
   * Enable one field
   * Enter SECRET_CH_ID in "secrets.h"
   * Enter SECRET_WRITE_APIKEY in "secrets.h"
 Setup Wi-Fi:
  * Enter SECRET_SSID in "secrets.h"
  * Enter SECRET_PASS in "secrets.h"
  
 Tutorial: http://nothans.com/measure-wi-fi-signal-levels-with-the-esp8266-and-thingspeak
   
 Created: Feb 1, 2017 by Hans Scharler (http://nothans.com)
*/

#include "ThingSpeak.h"


unsigned long myChannelNumber = 1077581;
const char * myWriteAPIKey = "Y18O4N0VWF7DYRA2";

#include <ESP8266WiFi.h>


char ssid[] = "private";   // your network SSID (name)
char pass[] = "private";   // your network password
int keyIndex = 0;            // your network key index number (needed only for WEP)
WiFiClient  client;
float Glucose_Values = random(70,170);



void setup() {
  Serial.begin(115200);

   WiFi.begin(ssid, pass);
  Serial.println("Connecting");
  while(WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());

  WiFi.mode(WIFI_STA);

  ThingSpeak.begin(client);
}

void loop() {

  // Connect or reconnect to WiFi
 
  // Measure Signal Strength (RSSI) of Wi-Fi connection
  long rssi = WiFi.RSSI();
  Glucose_Values = random(70,170); 
  
  
  // Write value to Field 1 of a ThingSpeak Channel
  int httpCode = ThingSpeak.writeField(myChannelNumber, 1, Glucose_Values, myWriteAPIKey);

  if (httpCode == 200) {
    Serial.println("Channel write successful.");
  }
  else {
    Serial.println("Problem writing to channel. HTTP error code " + String(httpCode));
  }

  // Wait 20 seconds to update the channel again
  delay(20000);
}
