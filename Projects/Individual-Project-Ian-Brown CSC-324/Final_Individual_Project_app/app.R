
library(tidyverse)
library(readr)
library(dplyr)
library(shiny)
library(shinydashboard)
library(lubridate)
library(ggplot2)
library(rsconnect)
library(plotly)


#Loading Data
indy <- read.csv("Ind_User_streaming_data_netflix.csv")
Titles <- read.csv("netflic_titles_project_data.csv")
log_data <- read.csv("Individual_Project_Log.csv")

# Cleaning data
Titles_Cleaned <- filter(Titles, country == "United Kingdom" & date_added > "1/1/2017" & type == "Movie")
indy_Cleaned <- filter(indy, watchTime > 600)
merged_data <- merge(Titles_Cleaned, indy_Cleaned, by = "title")

# color ramp
pubu <- RColorBrewer::brewer.pal(9, "PuBu")
col_p <- colorRampPalette(pubu)

# Making Calender Theme
theme_calendar <- function(){
    
    theme(aspect.ratio = 1/2, # Adjust as needed
          
          # Removes unessesary information from the plot
          axis.title = element_blank(),
          axis.ticks = element_blank(),
          axis.text.y = element_blank(),
          axis.text = element_text(family = "Montserrat"), 
          
          # Removes the grid lines and turns the background normal and blank. 
          panel.grid = element_blank(),
          panel.background = element_blank(),
          strip.background = element_blank(),
          
          # Sets the font, size, and size of the facet strip. 
          strip.text = element_text(family = "Montserrat", face = "bold", size = 15), 
          
          #Adjust Legend:
          legend.position = "top",
          legend.text = element_text(family = "Montserrat", hjust = .5),  
          legend.title = element_text(family = "Montserrat", size = 9, hjust = 1),  
          
          # Sets the possitioning of all the imagery of the heat map. 
          plot.caption = element_text(family = "Montserrat", hjust = 1, size = 8),  
          panel.border = element_rect(colour = "grey", fill=NA, linewidth = 1),
          plot.title = element_text(family = "Montserrat", hjust = .5, size = 26,  
                                    face = "bold", 
                                    margin = margin(0,0,0.5,0, unit = "cm")),
          plot.subtitle = element_text(family = "Montserrat", hjust = .5, size = 16)  
    )
}

ui <- dashboardPage(
    dashboardHeader(title = "Ian's Individual Project: Netflix Viewership"),
    ## Sidebar content
    dashboardSidebar(
        sidebarMenu(
            # First
            menuItem("BarGraph", tabName = "dashboard1", icon = icon("dashboard")),
            
            # Second
            menuItem("ScatterPlot", tabName = "dashboard2", icon = icon("dashboard")),
            
            # Third 
            menuItem("Week of the Year HeatMap", tabName = "dashboard3", icon = icon("dashboard")),
            
            #Fourth
            menuItem("Movie Popularity Over Time", tabName = "dashboard4", icon = icon("dashboard")), 
            
            #Fifth
            menuItem("Individual Project Log", tabName = "dashboard5", icon = icon("dashboard"))
        )
    ),
    ## Body content
    dashboardBody(
        tabItems(
            # First tab content: Bar Graph
            tabItem(tabName = "dashboard1",
                    # First app visualization:
                    bootstrapPage(
                        
                        plotOutput(outputId = "plot1", height = "400px"),
                    )
            ),
            
            # Second tab content: Scatter Plot
            tabItem(tabName = "dashboard2",
                    # Second app visualization:
                    fluidPage(
                        titlePanel("Movie Length vs. Watch Time"),
                        sidebarLayout(
                            sidebarPanel(
                                sliderInput("dateRange",
                                            "Select Date Range:",
                                            min = as.Date("2017-01-01"),
                                            max = as.Date("2019-06-30"),
                                            value = c(as.Date("2017-01-01"), as.Date("2019-06-30")),
                                            timeFormat = "%Y-%m-%d")
                            ),
                            mainPanel(
                                plotOutput("plot2")
                            )
                        )
                    )
                    
            ),
            
            # Third tab content: HeatMap
            tabItem(tabName = "dashboard3",
                    # Third app visualization:
                    fluidPage(
                        titlePanel("Calendar Heatmap of Movie Viewing"),
                        sidebarLayout(
                            sidebarPanel(
                                selectInput(inputId = "Year",
                                                     label = "Year Select:",
                                                     choices = c(2017, 2018, 2019),
                                                     selected = 2017),
                                ),
                            
                            mainPanel(
                                plotOutput("plot3", width = 1000, height = 1000)
                            )
                        )
                    )
            ),
            
            # Fourth tab content: Line Graph
            tabItem(tabName = "dashboard4",
                    # Fourth app visualization:
                    fluidPage(
                        
                        titlePanel("Movie Popularity Over Time"),
                        
                        # Output for the interactive plot
                        plotlyOutput("plot4", width = 1000, height = 1000) 
                        
                    )
            ), 
            
            #Fifth Tab Content: Iny Bar Graph
            tabItem(tabName = "dashboard5",
                    # fifth app visualization:
                    titlePanel("Work Time Distribution"),
                    mainPanel(
                        plotOutput("plot5")
                    )
            )
        )
    )
)


server <- function(input, output) {
    
    # First
    output$plot1 <- renderPlot({
        
        # Monthly histogram showing monthly viewership rates
        hist(month(ymd(indy_Cleaned$date)),
             # probability = TRUE,
             ylim = c(0, 100000),
             breaks = 12,
             xlab = "Viewership Frequency per Month",
             ylab = "# of Viewers", 
             main = "# of Viewers Every month")
        
    })
    
    # Second
    output$plot2 <- renderPlot({
        
        # Ensure that data is all in correct format. 
        merged_data$date <- as.Date(merged_data$date, "%Y-%m-%d") # Adjust the date format as necessary
        
        # Filter data based on the selected date range
        filtered_data <- merged_data[merged_data$date >= input$dateRange[1] & merged_data$date <= input$dateRange[2], ]
        
        # Create the scatter plot
        ggplot(filtered_data, aes(x = as.numeric(duration), y = as.numeric(watchTime) / 60 / as.numeric(duration))) +
            geom_point() +
            geom_smooth(na.rm = TRUE) +
            theme_minimal() +
            labs(x = "Movie Length", y = "Average Watch Time", title = "Movie Length vs. Average Watch Time") +
            theme(legend.position = "none") # Remove legend to keep plot clean
        
    })
    
    # Third
    output$plot3 <- renderPlot({
        # Convert date to a date format
        merged_data$date <- as.Date(merged_data$date)
        
        # Create the Month variable before subsetting or aggregating
        merged_data$Month <- month(merged_data$date, label = TRUE, abbr = FALSE)
        
        # Separate the data down to the selected year. 
        merged_data <- merged_data[year(merged_data$date) == input$Year, ]
        
        # Ensure the Month variable is included in the aggregated data frame
        merged_data_agg <- merged_data %>%
            group_by(date = floor_date(date, "day"), Month) %>%
            summarise(mean_watchTime = mean(watchTime, na.rm = TRUE))
        
        # Calculate the width of each bin
        # Find the minimum and maximum values
        min_value <- min(merged_data_agg$mean_watchTime, na.rm = TRUE)
        max_value <- max(merged_data_agg$mean_watchTime, na.rm = TRUE)
        
        # Calculate the width of each bin
        bin_width <- (max_value - min_value) / 9
        
        # Calculate the break points for the 9 bins
        breaks <- seq(min_value, max_value, by = bin_width)
        
        # Use the cut() function with explicit breaks
        merged_data_agg$mean_watchTime_discrete <- cut(merged_data_agg$mean_watchTime, breaks = breaks)
        
        # Heat-map Plot: This uses a calender display to show the users on a given day. 
        ggplot(merged_data_agg, 
               aes(wday(date, label = TRUE), -week(date), fill = mean_watchTime_discrete)) +
            geom_tile(color = "white", size = .4) +
            geom_text(aes(label = day(date), color = "black"), size = 2.5) +
            guides(fill = guide_colorsteps(barwidth = 25, 
                                           barheight = .4, 
                                           title.position = "top")) +
            scale_fill_manual(values = c("white", col_p(9)), 
                              na.value = "grey90", drop = FALSE) +
            scale_color_manual(values = c("black", "white"), guide = FALSE) + 
            facet_wrap(~ Month, nrow = 4, ncol = 3, scales = "free") +
            labs(title = "Average Daily Watch Time", 
                 subtitle = "Netflix Viewership",
                 caption = "",
                 fill = "# of Viewers") +
            theme_calendar()
        
        
    }, height = 600, width = 600)
    
    # Fourth Plot: Line Chart Showing watch times of each Individual Film. 
    output$plot4 <- renderPlotly({
        # Convert WatchDate to Date format
        merged_data$date <- as.Date(merged_data$date)
        
        # Aggregate data by MovieTitle and WatchDate, summing WatchDuration
        daily_data <- merged_data %>%
            group_by(title, date) %>%
            summarise(totalWatchTime = sum(watchTime)) %>%
            ungroup()
        
        # Line Chart output using plotly. This was used due to the convenience of having readily accessibly data for cursor hovering. 
        p <- plot_ly(daily_data, x = ~date, y = ~totalWatchTime, color = ~title, type = 'scatter', mode = 'lines') %>%
            layout(title = "Movie Popularity Over Time",
                   xaxis = list(title = "Date"),
                   yaxis = list(title = "Total Duration Watched"),
                   legend = list(orientation = "h", y = -0.15, x = 0.5))
        
        p
    })
    
    # Fifth Plot: Bar Graph of Individual Time Spent
    output$plot5 <- renderPlot({
        # Aggregating data for summary usage:
        agg_log_data <- log_data %>%
            group_by(Activity) %>%
            summarise(TotalTime = sum(Total_time)) %>%
            ungroup()
        
        # Calculate total time for all work types:
        totalTime <- sum(agg_log_data$TotalTime)
            
        # Calculate percentage contribution of each work type:
        agg_log_data$Percentage <- agg_log_data$TotalTime / totalTime * 100
        
        # Creating the bar chart for the descriptive data: 
        ggplot(agg_log_data, aes(x = Activity, y = TotalTime)) +
            geom_bar(stat = "identity", fill = "steelblue") +
            theme_minimal() +
            labs(x = "Activity", y = "Total Time", title = "Work Time Distribution")
        
        })
}

shinyApp(ui, server)



