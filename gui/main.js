const electron = require("electron");
const url = require("url");
const path = require("path");
const { Main } = require("electron");

const { app, BrowserWindow, Menu, ipcMain } = electron;


let mainWindow, addWindow;
let chat = [];
let sendername="emre";

app.on('ready', () => {
    mainWindow = new BrowserWindow({
   
        frame: false,
        webPreferences: {
            nodeIntegration: true,

        }});
    mainWindow.loadURL(
        url.format({
            pathname: path.join(__dirname, "start.html"),
            protocol: "file:",
            slashes: true
            
        }));

    const mainMenu = Menu.buildFromTemplate(mainMenuTemplate);
    Menu.setApplicationMenu(mainMenu);

//  mainWindow.setResizable(false);
    ipcMain.on("NewCamera", (err, data) => {
        console.log("Trigger.");
        // Burada Aslında Back-End de 'biri kamera açtı' bildiriminin front ende
        // aktarılmasını test ediyoruz.
        // Şimdilik back-endi tetiklemek için sendbtn'nin fronttan bir tıklama yapmasını bekliyoruz.
        // Normalde back -> front olacak. Front -> back -> front yapıyoruz.

        addWindow.webContents.send("key:Someone" , "AnyPerson");
    })
    ipcMain.on("key:maximize", (err, data) => {
        const a = BrowserWindow.getFocusedWindow();
        if (a.isMaximized()) {

            a.unmaximize();
            console.log("unmax");
           
        } else {
            a.maximize();
            console.log("max");
        }
       

    })
    ipcMain.on("key:minimize", () => {
        const a = BrowserWindow.getFocusedWindow();
        a.minimize();

    })
    ipcMain.on("key:closeBtn", () => {
        const a = BrowserWindow.getFocusedWindow();
        a.close();


    })
    mainWindow.on('close', () => {
        app.quit();

    })
    ipcMain.on("key:opencall", () => {
       
        lessonscreen();


    })
    ipcMain.on("key:join", () => {

        createWindow();


    })
    ipcMain.on("key:login", () => {
       
        loginscreen();


    })
    ipcMain.on("key:calendar", () => {

        calendarscreen();


    })
    ipcMain.on("chatInput", (err, data) => {
        if (data) {
            chat.push(
                {
                    id: sendername,
                    text: data
                });
            addWindow.webContents.send("showchat",chat);
            getChat();
        }

    })
})

const mainMenuTemplate = [{
    label: "Dosya",
    submenu: [{
        label: "Selam"
    },
    {
        label: "naber"
    },
    {
        label: "��k��",
        role: "quit"
    },
        {
            label: "devtools",
            click(item, focusedWindow) {
                focusedWindow.toggleDevTools();
            }
        }]
    

}]

function createWindow() {

    addWindow = new BrowserWindow({
        webPreferences: {
            nodeIntegration: true,
           
},
        width: 1440,
        height: 960,
        title: "patates",
        frame:false,
})
    addWindow.loadURL(url.format({
        pathname: path.join(__dirname, "main.html"),
        protocol: "file:",
        slashes: true,

        
    }))
    addWindow.on('close', () => {
        addWindow = null;
    })
}
function loginscreen() {

    logWindow = new BrowserWindow({
        webPreferences: {
            nodeIntegration: true
        },
        width: 480,
        height: 580,
        title: "login",
        frame: false,
    })
    logWindow.loadURL(url.format({
        pathname: path.join(__dirname, "login.html"),
        protocol: "file:",
        slashes: true,

    }))
    logWindow.setResizable(false);
    logWindow.on('close', () => {
        logWindow = null;
    })
}
function lessonscreen() {

    lessonWindow = new BrowserWindow({
        webPreferences: {
            nodeIntegration: true
        },
        width: 640,
        height: 780,
        title: "login",
        frame: false,
    })
    lessonWindow.loadURL(url.format({
        pathname: path.join(__dirname, "lesson.html"),
        protocol: "file:",
        slashes: true,

    }))
    lessonWindow.setResizable(false);
    lessonWindow.on('close', () => {
        lessonWindow = null;
    })
}
function calendarscreen() {

    calendarWindow = new BrowserWindow({
        webPreferences: {
            nodeIntegration: true
        },
        width: 610,
        height: 810,
        title: "login",
        frame: false,
    })
    calendarWindow.loadURL(url.format({
        pathname: path.join(__dirname, "calendar.html"),
        protocol: "file:",
        slashes: true,

    }))
    calendarWindow.setResizable(false);
    calendarWindow.on('close', () => {
        calendarWindow = null;
    })
}
function getChat() {
    console.log(chat);


}