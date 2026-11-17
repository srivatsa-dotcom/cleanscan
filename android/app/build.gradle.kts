plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.cleanscan.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.cleanscan.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        // segmentation model in src/main/assets (YOLO-seg layout, COCO classes); the bundled
        // living_seg.tflite is Ultralytics YOLO11n-seg (AGPL-3.0), see LICENSES.md.
        // Build with another one: ./gradlew assembleDebug -Pcleanscan.segModel=my_model.tflite
        manifestPlaceholders["segModel"] = (project.findProperty("cleanscan.segModel") ?: "living_seg.tflite").toString()
    }

    // CI signs release builds when a keystore is given (.github/workflows/android.yml)
    val keystore = System.getenv("CLEANSCAN_KEYSTORE")
    signingConfigs {
        if (keystore != null) {
            create("ci") {
                storeFile = file(keystore)
                storePassword = System.getenv("CLEANSCAN_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("CLEANSCAN_KEY_ALIAS")
                keyPassword = System.getenv("CLEANSCAN_KEY_PASSWORD")
            }
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            if (keystore != null) signingConfig = signingConfigs.getByName("ci")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    androidResources {
        noCompress += "tflite"  // the model is memory-mapped
    }
}

dependencies {
    implementation("com.google.ar:core:1.45.0")
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.core:core-ktx:1.13.1")
}
