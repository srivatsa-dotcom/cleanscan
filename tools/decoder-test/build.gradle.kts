// Runs the Android app's YoloSegDecoder.kt on the plain JVM (no Android SDK needed),
// driven by tools/verify_android_model.py.
plugins {
    kotlin("jvm") version "1.9.24"
    application
}
repositories { mavenCentral() }
sourceSets["main"].kotlin {
    srcDir("../../android/app/src/main/java")
    include("**/YoloSegDecoder.kt", "**/Main.kt")
}
application { mainClass.set("MainKt") }
